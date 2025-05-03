import sys
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse
import sglang.srt.entrypoints.http_server as server
from sglang.srt.entrypoints.http_server import app, _global_state, set_global_state, launch_server
from sglang.srt.server_args import prepare_server_args, ServerArgs
from sglang.srt.openai_api.adapter import v1_completions, v1_chat_completions
from dynasor.cli.sglang_adapter import adaptive_v1_completions, get_chat_completion_prompt
from dynasor.core.entropy import obtain_answer, should_early_exit, uncertain_words, is_certain_answer
import logging

logger = logging.getLogger(__name__)

async def handle_adaptive_compute(raw_request: Request, request):

    original_request = request.copy()
    adaptive_compute = request["adaptive_compute"]

    # Setup basic configurations (will be updated)
    remaining_tokens = request.get("max_tokens", 2048)#confirm this TODO

    sending_prompt = request["prompt"]
    answers = []
    is_certains = []
    output_var_template: Optional[CompletionStreamResponse] = None

    # Setup basic constants
    probe_text = adaptive_compute.get("probe_text", None)
    probe_text_end = adaptive_compute.get("probe_text_end", None)
    certainty_window = adaptive_compute.get("certainty_window", 2)
    token_interval = adaptive_compute.get("token_interval", 32)

    assert probe_text is not None

    # Coordination state - for stop condition
    certainty_event = asyncio.Event()
    final_answer_box = {"answer": None}

    async def launch_probe(prompt_snapshot: str):
        probe_prompt = prompt_snapshot + probe_text
        probe_request = original_request.model_copy()
        probe_request.prompt = probe_prompt
        probe_request.stream = True
        probe_request.max_tokens = 20
        probe_request.temperature = 0.6
        probe_request.top_p = 0.95
        probe_request.adaptive_compute = None

        try:
            probe_generator = await adaptive_v1_completions(_global_state.tokenizer_manager,probe_request, raw_request)
            probe_output_text = ""
            async for output in probe_generator:
                token = output.choices[0].text
                probe_output_text += token
            answer = obtain_answer(probe_output_text)
            is_certain = is_certain_answer(probe_output_text, uncertain_words)
            answers.append(answer)
            is_certains.append(is_certain)
            if should_early_exit(answers, probe_output_text, uncertain_words, certainty_window, is_certains):
                final_answer_box["answer"] = answer
                certainty_event.set()
        except Exception as e:
            logger.error(f"Probe task failed: {e}")

    # Setup the request for actual query.
    while remaining_tokens > 0:
        decoding_request = original_request.copy()
        decoding_request.prompt = sending_prompt
        decoding_request.stream = True
        decoding_request.max_tokens = min(token_interval, remaining_tokens)
        decoding_request.adaptive_compute = None
        generator = await adaptive_v1_completions(_global_state.tokenizer_manager,decoding_request, raw_request)
        if isinstance(generator, ErrorResponse):
            error_response = generator.model_dump()
            yield f"data: {error_response}\n\n"
            yield "data: [DONE]\n\n"
            return

        token_buffer = ""
        token_count = 0

        # Send the actual query
        async for output in generator:
            if certainty_event.is_set():
                break  # Stop if model was confident

            token = output.choices[0].text
            token_buffer += token
            remaining_tokens -= 1
            token_count += 1

            response_json = output.model_dump_json(exclude_unset=False, exclude_none=True)
            if output_var_template is None:
                output_var_template = output

            yield f"data: {response_json}\n\n"

            if token_count >= token_interval:
                # Launch probe task asynchronously
                asyncio.create_task(launch_probe(sending_prompt + token_buffer))
                token_count = 0

            if remaining_tokens <= 0:
                break

        sending_prompt += token_buffer

        if certainty_event.is_set(): #early exit condition is met
            answer = final_answer_box["answer"]
            logger.debug(f"Early exit on answer: {answer = }")

            response_obj = output_var_template.model_copy()
            response_obj.choices[0].text = probe_text + answer + probe_text_end
            response_obj.choices[0].finish_reason = "stop"
            response_obj.choices[0].stop_reason = "stop"

            response_json = response_obj.model_dump_json(
                exclude_unset=False, exclude_none=True
            )
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"
            return

        if remaining_tokens <= 0:
            logger.debug(f"Early exit: Remaining tokens <= 0")
            yield f"data: [DONE]\n\n"
            break

    yield "data: [DONE]\n\n"

app.router.routes[:] = [
    route
    for route in app.router.routes
    if route.path not in ("/v1/completions", "/v1/chat/completions")
]


@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    json_obj = await raw_request.json()
    adaptive_compute = json_obj.get("adaptive_compute", None)
    if adaptive_compute is not None:
        generator = handle_adaptive_compute(raw_request,json_obj)
        return StreamingResponse(content=generator, media_type="text/event-stream")
    return await v1_completions(_global_state.tokenizer_manager, raw_request)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    json_obj = await raw_request.json()
    adaptive_compute = json_obj.get("adaptive_compute", None)
    if adaptive_compute is not None:
        json_obj["prompt"]=get_chat_completion_prompt(json_obj,_global_state.tokenizer_manager)
        generator = handle_adaptive_compute(raw_request,json_obj)
        return StreamingResponse(content=generator, media_type="text/event-stream")
    return await v1_chat_completions(_global_state.tokenizer_manager, raw_request)

def main():
    server_args = prepare_server_args(sys.argv[1:])
    try: 
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

if __name__ == "__main__":
    main()
    