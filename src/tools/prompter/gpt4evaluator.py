from .openai import OpenAIMetaPrompter


class OpenAIBinsumEvaluator(OpenAIMetaPrompter):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        self.prompt = """
You are an experienced C/C++ software developer. Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistants to summarize the functionality of a piece of decompiled code.  For each query, the user will provide the decompiled code, the corresponding source code, and a reference summary is also provided.
In the evaluation, you should answer the following questions:

A. Does the summary reflect relevant context (domain)? Answer the question in range 5(best) to 1(worst).
Domain/context describes the purpose of a function. It is more of the general high-level domain (e.g., network, memory, CPS, physics, GUI, etc) rather than specific functionalities (e.g., sort, string comparison, memory allocation).
For 5, the summary and the reference should describe the same domain/context.
For 4, the domain of the summary and the reference should be similar and relevant, although may not be exactly the same. The summary domain may be a superset or subset of the reference. The summary domain may be closely related to the reference domain. The summary and reference may be two different perspectives of a same specific domain.
For 3, the summary does not explicitly mention a specific context. It only contains low level operations. From the summary, one cannot deduce the high-level purpose of the decompiled function.
For 2, the summary is slightly misleading. The summary domain is different and not relevant to the reference domain. However, it is implied by the choice of words in the summary, and is not explicitly mentioned.
For 1, the summary is completely misleading. The summary domain is irrelevant to the reference domain, and it is explicitly mentioned in the summary.

B. Does the summary reflect relevant functionality? Answer the question in range 1(best) to 5(worst).
Functionality means the specific high-level behaviors performed in a function (e.g., sort, string comparison, decoding package, printing error messages).
For 5, the functionality in the summary should be almost exactly the same to the reference.
For 4, the functionalities in the summary are similar to the reference. It may be vague in details, but the overall functionality and purpose is correct.
For 3, the summary does not specify functionality. It only repeats some low-level operations without high level abstractions.
For 2, the summary specify relevant but inaccurate functionality. The functionality specified in the summary may be relevant to the reference summary, but they have significant differences.
For 1, the summary contains irrelevant functionality. It is contains a totally different behavior with the reference.

Your output should first briefly comment the summary from the aforementioned perspectives, then answer the questions in the following format:
```score
{'Q-A': [1-5], 'Q-B': [1-5]}
```

Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
"""
    def pack_query(
        self,
        decompiled, 
        src, 
        srcsum, 
        pred_sum
    ):
        ret = ""
        ret += "// Decompiled code: \n \"\"\"\n%s\n\"\"\"\n\n" % decompiled
        ret += "// Source code: \n \"\"\"\n%s\n\"\"\"\n\n" % src
        ret += "// Reference summary: \n \"\"\"\n%s\n\"\"\"\n\n" % srcsum
        ret += "// Queried summary: \n %s\n" % pred_sum
        return ret

    def evaluate(
        self,
        query,
        **kwargs
    ) -> dict:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": self.pack_query(query[0], query[1], query[2], query[3])},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            seed=42,
            **kwargs
        )
        return response
