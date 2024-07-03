import google.generativeai as genai
from typing import List, Tuple


class GenaiMetaPrompter(object):

    def __init__(
        self,
        api_key: str,
        model_name: str,
    ):
        
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]


class GenaiSourceSummarizer(GenaiMetaPrompter):

    def __init__(
        self, 
        api_key: str, 
        model_name: str
    ):
        super().__init__(api_key, model_name)
        self.system_prompt = \
        "You are an experienced C/C++ software developer. "
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            # system_instruction=self.system_prompt
        )
        self.user_prompt = \
"""You are provided with the following function:
```C/C++
{}
```
First generate a brief step-by-step description of its functionality in the format:
**Description**:
...

Then generate a high-level summary of its functionality in the format:
**Summary**:
The function ...

After that, generate a brief description of its general purpose in the format:
**Purpose**:
The purpose of the function is to ...

"""
    def pack_src(
        self,
        source_function
    ):
        return self.system_prompt + \
            self.user_prompt.format(source_function)

    def generate(
        self,
        inputs,
        generation_config
    ):
        return self.model.generate_content(
            contents=self.pack_src(inputs),
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )
    

class GenaiDecompSummarizer(GenaiMetaPrompter):

    def __init__(
        self, 
        api_key: str, 
        model_name: str
    ):
        super().__init__(api_key, model_name)

        self.system_prompt = \
        "You are an experienced binary reverse engineer to understand "\
        "decompiled C code that lacks symbol information. "
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            # system_instruction=self.system_prompt
        )

        self.default_user_prompt = \
"""You are provided with the following decompiled function that is hardly human readable:
```C
{}
```
First generate a brief step-by-step description of its functionality in the format:
**Description**:
...

Then try to generate a summary of it that can help human understand / inspect its original high-level source code functionality in the format:
**Summary**:
The function ...

After that, inspect and generate a brief description of its general purpose in the format:
**Purpose**:
The purpose of the function seems to ...
"""
        self.augmented_user_prompt = \
"""You are provided with the following decompiled function that is not human readable:
```C
{}
```

First generate a brief step-by-step description of the functionality of the decompiled code in the format:
**Description**:
...

Then try to generate a summary of it that can help human understand / inspect its original high-level source code functionality in the format:
**Summary**:
The function ...

After that, consider the following source code fragments (might not be complete function) that are potentially relevant to the this decompiled function.
{}

Analyze whether they are relevant to the decompiled function in the format:
**Analysis**:
...

Finally, based on the analysis, try to inspect and generate the general purpose of the decompiled function in the format:
**Purpose**:
The purpose of the function seems to ...
"""
# Finally, given relevant source code information, try to inspect and generate the general purpose of the decompiled function in the format:

    def pack_dec(
        self,
        decompiled_function,
    ):
        return self.system_prompt + \
            self.default_user_prompt.format(decompiled_function)
    
    def pack_dec_w_aug(
        self,
        decompiled_function: str,
        source_candidates: List[str],
    ):
        content = self.system_prompt + \
            self.augmented_user_prompt.format(
                decompiled_function,
                '\n'.join(
                [f'Potential source function {i+1}:\n```{s.rstrip()}\n```' \
                    for i, s in enumerate(source_candidates)])
            )
        return content

    def generate(
        self,
        inputs,
        generation_config
    ):
        return self.model.generate_content(
            contents=self.pack_dec(inputs),
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )

    def generate_with_augmentation(
        self,
        inputs,
        generation_config
    ):
        assert len(inputs) == 2
        return self.model.generate_content(
            contents=self.pack_dec_w_aug(inputs[0], inputs[1]),
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )


class GenaiDecompFuncNamer(GenaiMetaPrompter):

    def __init__(
        self, 
        api_key: str, 
        model_name: str
    ):
        super().__init__(api_key, model_name)
        self.system_prompt = \
        "You are an experienced binary reverse engineer to understand "\
        "decompiled C code that lacks symbol information. "
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            # system_instruction=self.system_prompt
        )

        self.default_user_prompt = \
"""You have decompiled a function from an executable, which currently has a generic name like `sub_xxx`. The decompiled function code is as follows:
```C
{}
```
Generate a more human-understandable function name for the decompiled code to replace the original `sub_xxx` in the format:
**Function Name**: `function_name_goes_here`
"""
        self.augmented_user_prompt = \
"""You have decompiled a function from an executable, which currently has a generic name like `sub_xxx`. The decompiled function code is as follows:
```C
{}
```

Consider the following source code fragments (might not be complete function) that are potentially relevant to the this decompiled function.
{}

Analyze whether these source functions are relevant to the decompiled function in the format:
**Analysis**:
...

Then, based on the analysis, generate a more human-understandable function name for the decompiled code to replace the original `sub_xxx` in the format:
**Function Name**:  `function_name_goes_here`
"""
# Then, properly leverage the naming information in the relevant source functions based on the analysis, and generate a more human-understandable function name for the decompiled code to replace the original `sub_xxx` in the format:

    def pack_dec(
        self,
        decompiled_function,
    ):
        # return self.default_user_prompt.format(decompiled_function)
        return self.system_prompt + \
            self.default_user_prompt.format(decompiled_function)
    
    def pack_dec_w_aug(
        self,
        decompiled_function: str,
        source_candidates: List[str],
    ):
        content = self.system_prompt + \
            self.augmented_user_prompt.format(
                decompiled_function,
                '\n'.join(
                [f'Potential source function {i+1}:\n```{s.rstrip()}\n```' \
                    for i, s in enumerate(source_candidates)])
            )
        return content

    def generate(
        self,
        inputs,
        generation_config
    ):
        return self.model.generate_content(
            contents=self.pack_dec(inputs),
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )

    def generate_with_augmentation(
        self,
        inputs,
        generation_config
    ):
        assert len(inputs) == 2
        return self.model.generate_content(
            contents=self.pack_dec_w_aug(inputs[0], inputs[1]),
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )