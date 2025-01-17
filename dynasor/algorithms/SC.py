from dynasor.algorithms.Reasoning import ReasoningProgram

class SelfConsistency(ReasoningProgram):
    def __init__(self, input_message: dict):
        super().__init__()
        self.sampling_params = dict(
            max_tokens=512,
            temperature=0.7,
        )
        self.input_message = input_message

    def init_resource(self, app: Client):
        self.knobs['n_branches'] = 5
        pass

    def expand(self, app: Client):
        knobs = self.knobs
        n_branches = knobs['n_branches']

        messages = self.input_message
        responses = app.get_responses(
            messages, n=n_branches,
            **self.sampling_params,
        )
        answers = [extract_answer(r) for r in responses]
        self.results.extend(answers)
        return

    def aggregate(self):
        results = self.results
        return majority_vote(results)
    
    def update_certitude(self):
        self.state['terminated'] = True
        return