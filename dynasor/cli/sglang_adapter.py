import json
import os
import time
from typing import Dict, List

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from sglang.srt.openai_api.adapter import v1_generate_request, v1_generate_response
from sglang.srt.openai_api.protocol import (
    BatchRequest,
    BatchResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    FileDeleteResponse,
    FileRequest,
    FileResponse,
    FunctionResponse,
    LogProbs,
    MultimodalEmbeddingInput,
    ToolCall,
    TopLogprob,
    UsageInfo,
)


chat_template_name = None

async def adaptive_v1_completions(tokenizer_manager,request_json, raw_request: Request):

    all_requests = [CompletionRequest(**request_json)]
    created = int(time.time())
    adapted_request, request = v1_generate_request(all_requests)

    if adapted_request.stream:

        async def generate_stream_resp():
            stream_buffers = {}
            n_prev_tokens = {}
            prompt_tokens = {}
            completion_tokens = {}
            cached_tokens = {}

            try:
                async for content in tokenizer_manager.generate_request(
                    adapted_request, raw_request
                ):
                    index = content.get("index", 0)

                    stream_buffer = stream_buffers.get(index, "")
                    n_prev_token = n_prev_tokens.get(index, 0)

                    text = content["text"]
                    prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                    completion_tokens[index] = content["meta_info"]["completion_tokens"]
                    cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)

                    if not stream_buffer:  # The first chunk
                        if request.echo:
                            if isinstance(request.prompt, str):
                                # for the case of single str prompts
                                prompts = request.prompt
                            elif isinstance(request.prompt, list):
                                if isinstance(request.prompt[0], str):
                                    # for the case of multiple str prompts
                                    prompts = request.prompt[index // request.n]
                                elif isinstance(request.prompt[0], int):
                                    # for the case of single token ids prompt
                                    prompts = tokenizer_manager.tokenizer.decode(
                                        request.prompt, skip_special_tokens=True
                                    )
                                elif isinstance(request.prompt[0], list) and isinstance(
                                    request.prompt[0][0], int
                                ):
                                    # for the case of multiple token ids prompts
                                    prompts = tokenizer_manager.tokenizer.decode(
                                        request.prompt[index // request.n],
                                        skip_special_tokens=True,
                                    )

                            # Prepend prompt in response text.
                            text = prompts + text

                    if request.logprobs is not None:
                        # The first chunk and echo is enabled.
                        if not stream_buffer and request.echo:
                            input_token_logprobs = content["meta_info"][
                                "input_token_logprobs"
                            ]
                            input_top_logprobs = content["meta_info"][
                                "input_top_logprobs"
                            ]
                        else:
                            input_token_logprobs = None
                            input_top_logprobs = None

                        logprobs = to_openai_style_logprobs(
                            input_token_logprobs=input_token_logprobs,
                            input_top_logprobs=input_top_logprobs,
                            output_token_logprobs=content["meta_info"][
                                "output_token_logprobs"
                            ][n_prev_token:],
                            output_top_logprobs=content["meta_info"][
                                "output_top_logprobs"
                            ][n_prev_token:],
                        )
                        n_prev_token = len(
                            content["meta_info"]["output_token_logprobs"]
                        )
                    else:
                        logprobs = None

                    delta = text[len(stream_buffer) :]
                    stream_buffer = stream_buffer + delta
                    finish_reason = content["meta_info"]["finish_reason"]
                    choice_data = CompletionResponseStreamChoice(
                        index=index,
                        text=delta,
                        logprobs=logprobs,
                        finish_reason=finish_reason["type"] if finish_reason else None,
                        matched_stop=(
                            finish_reason["matched"]
                            if finish_reason and "matched" in finish_reason
                            else None
                        ),
                    )
                    chunk = CompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=created,
                        object="text_completion",
                        choices=[choice_data],
                        model=request.model,
                    )

                    stream_buffers[index] = stream_buffer
                    n_prev_tokens[index] = n_prev_token

                    yield chunk#f"data: {chunk.model_dump_json()}\n\n"
                if request.stream_options and request.stream_options.include_usage:
                    total_prompt_tokens = sum(
                        tokens
                        for i, tokens in prompt_tokens.items()
                        if i % request.n == 0
                    )
                    total_completion_tokens = sum(
                        tokens for tokens in completion_tokens.values()
                    )
                    cache_report = tokenizer_manager.server_args.enable_cache_report
                    if cache_report:
                        cached_tokens_sum = sum(
                            tokens for tokens in cached_tokens.values()
                        )
                        prompt_tokens_details = {"cached_tokens": cached_tokens_sum}
                    else:
                        prompt_tokens_details = None
                    usage = UsageInfo(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        prompt_tokens_details=prompt_tokens_details,
                    )

                    final_usage_chunk = CompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=created,
                        choices=[],
                        model=request.model,
                        usage=usage,
                    )
                    final_usage_data = final_usage_chunk.model_dump_json(
                        exclude_none=True
                    )
                    yield final_usage_chunk#f"data: {final_usage_data}\n\n"
            except ValueError as e:
                error = create_streaming_error_response(str(e))
                import json
                yield json.loads(error)
            #yield "data: [DONE]\n\n"


def get_chat_completion_prompt(
    request_json,
    tokenizer_manager
):
    request = ChatCompletionRequest(**request_json)
    prompt = ""
    # NOTE: with openai API, the prompt's logprobs are always not computed
        # Prep the data needed for the underlying GenerateReqInput:
        #  - prompt: The full prompt string.
        #  - stop: Custom stop tokens.
        #  - image_data: None or a list of image strings (URLs or base64 strings).
        #    None skips any image processing in GenerateReqInput.
    if not isinstance(request.messages, str):
        # Apply chat template and its stop strings.
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.function.model_dump()
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [item.function.model_dump() for item in request.tools]

        if chat_template_name is None:
            openai_compatible_messages = []
            for message in request.messages:
                if isinstance(message.content, str):
                    openai_compatible_messages.append(
                        {"role": message.role, "content": message.content}
                    )
                else:
                    content_list = message.dict()["content"]
                    for content in content_list:
                        if content["type"] == "text":
                            openai_compatible_messages.append(
                                {"role": message.role, "content": content["text"]}
                            )
            if openai_compatible_messages[-1]["role"] == "assistant":
                assistant_prefix = openai_compatible_messages[-1]["content"]
                openai_compatible_messages = openai_compatible_messages[:-1]
            else:
                assistant_prefix = None
            try:
                prompt = tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize= False  ,
                    add_generation_prompt=True,
                    tools=tools,
                )
            except:
                #  This except branch will be triggered when the chosen model
                #  has a different tools input format that is not compatiable
                #  with openAI's apply_chat_template tool_call format, like Mistral.
                tools = [t if "function" in t else {"function": t} for t in tools]
                prompt = tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=tools,
                )

        else:
            conv = generate_chat_conv(request, chat_template_name)
            prompt = conv.get_prompt()
    else:
        # Use the raw prompt and stop strings if the messages is already a string.
        prompt = request.messages
    return prompt

