# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
import json
from tqdm import tqdm
import torch

# import numpy  # for gradio hot reload
import gradio as gr

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LogitsProcessorList

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from evaluate import load
import numpy as np
import math
from sklearn import metrics
import google.generativeai as genai
import jsonlines

prompts = [
    "paraphrase the following paragraphs:\n",
    "“paraphrase the following paragraphs and try your best not to use the same bigrams from the original paragraphs:\n",
    "paraphrase the following paragraphs and try to keep the similar length to the original paragraphs:\n",
    "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n",
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases: \n",
]

safety_settings = [
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

genai.configure(api_key="AIzaSyAcL7v7OLOcBj4quiWdB_sMQxffVOXNltE")
model = genai.GenerativeModel(model_name="gemini-pro", safety_settings=safety_settings)


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API"
    )

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args


def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device


def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
    and generate watermarked text by passing it to the generate method of the model
    as a logits processor."""

    # print(f"Generating with {args}")

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
    )

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))
    else:
        gen_kwargs.update(dict(num_beams=args.n_beams))

    generate_without_watermark = partial(model.generate, **gen_kwargs)
    generate_with_watermark = partial(model.generate, logits_processor=LogitsProcessorList([watermark_processor]), **gen_kwargs)
    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048 - args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(
        device
    )
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, tokd_input["input_ids"].shape[-1] :]
        output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1] :]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input, int(truncation_warning), decoded_output_without_watermark, decoded_output_with_watermark, args)
    # decoded_output_with_watermark)


def format_names(s):
    """Format names for the gradio demo interface"""
    s = s.replace("num_tokens_scored", "Tokens Counted (T)")
    s = s.replace("num_green_tokens", "# Tokens in Greenlist")
    s = s.replace("green_fraction", "Fraction of T in Greenlist")
    s = s.replace("z_score", "z-score")
    s = s.replace("p_value", "p value")
    s = s.replace("prediction", "Prediction")
    s = s.replace("confidence", "Confidence")
    return s


def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k, v in score_dict.items():
        if k == "green_fraction":
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k == "confidence":
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float):
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else:
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2, ["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1, ["z-score Threshold", f"{detection_threshold}"])
    return lst_2d


def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
    the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_ngrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens,
    )
    if len(input_text) - 1 > 4:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error", "string too short to compute metrics"]]
        output += [["", ""] for _ in range(6)]
    return output, args


def run_gradio(args, model=None, device=None, tokenizer=None):
    """Define and launch the gradio demo interface"""
    generate_partial = partial(generate, model=model, device=device, tokenizer=tokenizer)
    detect_partial = partial(detect, device=device, tokenizer=tokenizer)

    with gr.Blocks() as demo:
        # Top section, greeting and instructions
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                    """
                ## 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 🔍
                """
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                [![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/jwkirchenbauer/lm-watermarking)
                """
                )
            # with gr.Column(scale=2):
            #     pass
            # ![visitor badge](https://visitor-badge.glitch.me/badge?page_id=tomg-group-umd_lm-watermarking) # buggy

        with gr.Accordion("Understanding the output metrics", open=False):
            gr.Markdown(
                """
            - `z-score threshold` : The cuttoff for the hypothesis test
            - `Tokens Counted (T)` : The number of tokens in the output that were counted by the detection algorithm. 
                The first token is ommitted in the simple, single token seeding scheme since there is no way to generate
                a greenlist for it as it has no prefix token(s). Under the "Ignore Bigram Repeats" detection algorithm, 
                described in the bottom panel, this can be much less than the total number of tokens generated if there is a lot of repetition.
            - `# Tokens in Greenlist` : The number of tokens that were observed to fall in their respective greenlist
            - `Fraction of T in Greenlist` : The `# Tokens in Greenlist` / `T`. This is expected to be approximately `gamma` for human/unwatermarked text.
            - `z-score` : The test statistic for the detection hypothesis test. If larger than the `z-score threshold` 
                we "reject the null hypothesis" that the text is human/unwatermarked, and conclude it is watermarked
            - `p value` : The likelihood of observing the computed `z-score` under the null hypothesis. This is the likelihood of 
                observing the `Fraction of T in Greenlist` given that the text was generated without knowledge of the watermark procedure/greenlists.
                If this is extremely _small_ we are confident that this many green tokens was not chosen by random chance.
            -  `prediction` : The outcome of the hypothesis test - whether the observed `z-score` was higher than the `z-score threshold`
            - `confidence` : If we reject the null hypothesis, and the `prediction` is "Watermarked", then we report 1-`p value` to represent 
                the confidence of the detection based on the unlikeliness of this `z-score` observation.
            """
            )

        with gr.Accordion("A note on model capability", open=True):
            gr.Markdown(
                """
                This demo uses open-source language models that fit on a single GPU. These models are less powerful than proprietary commercial tools like ChatGPT, Claude, or Bard. 

                Importantly, we use a language model that is designed to "complete" your prompt, and not a model this is fine-tuned to follow instructions. 
                For best results, prompt the model with a few sentences that form the beginning of a paragraph, and then allow it to "continue" your paragraph. 
                Some examples include the opening paragraph of a wikipedia article, or the first few sentences of a story. 
                Longer prompts that end mid-sentence will result in more fluent generations.
                """
            )
        gr.Markdown(f"Language model: {args.model_name_or_path} {'(float16 mode)' if args.load_fp16 else ''}")

        # Construct state for parameters, define updates and toggles
        default_prompt = args.__dict__.pop("default_prompt")
        session_args = gr.State(value=args)

        with gr.Tab("Generate and Detect"):

            with gr.Row():
                prompt = gr.Textbox(label=f"Prompt", interactive=True, lines=10, max_lines=10, value=default_prompt)
            with gr.Row():
                generate_btn = gr.Button("Generate")
            with gr.Row():
                with gr.Column(scale=2):
                    output_without_watermark = gr.Textbox(label="Output Without Watermark", interactive=False, lines=14, max_lines=14)
                with gr.Column(scale=1):
                    # without_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    without_watermark_detection_result = gr.Dataframe(
                        headers=["Metric", "Value"], interactive=False, row_count=7, col_count=2
                    )
            with gr.Row():
                with gr.Column(scale=2):
                    output_with_watermark = gr.Textbox(label="Output With Watermark", interactive=False, lines=14, max_lines=14)
                with gr.Column(scale=1):
                    # with_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    with_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False, row_count=7, col_count=2)

            redecoded_input = gr.Textbox(visible=False)
            truncation_warning = gr.Number(visible=False)

            def truncate_prompt(redecoded_input, truncation_warning, orig_prompt, args):
                if truncation_warning:
                    return redecoded_input + f"\n\n[Prompt was truncated before generation due to length...]", args
                else:
                    return orig_prompt, args

        with gr.Tab("Detector Only"):
            with gr.Row():
                with gr.Column(scale=2):
                    detection_input = gr.Textbox(label="Text to Analyze", interactive=True, lines=14, max_lines=14)
                with gr.Column(scale=1):
                    # detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False, row_count=7, col_count=2)
            with gr.Row():
                detect_btn = gr.Button("Detect")

        # Parameter selection group
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"#### Generation Parameters")
                    with gr.Row():
                        decoding = gr.Radio(
                            label="Decoding Method",
                            choices=["multinomial", "greedy"],
                            value=("multinomial" if args.use_sampling else "greedy"),
                        )
                    with gr.Row():
                        sampling_temp = gr.Slider(
                            label="Sampling Temperature", minimum=0.1, maximum=1.0, step=0.1, value=args.sampling_temp, visible=True
                        )
                    with gr.Row():
                        generation_seed = gr.Number(label="Generation Seed", value=args.generation_seed, interactive=True)
                    with gr.Row():
                        n_beams = gr.Dropdown(
                            label="Number of Beams", choices=list(range(1, 11, 1)), value=args.n_beams, visible=(not args.use_sampling)
                        )
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            label="Max Generated Tokens", minimum=10, maximum=1000, step=10, value=args.max_new_tokens
                        )

                with gr.Column(scale=1):
                    gr.Markdown(f"#### Watermark Parameters")
                    with gr.Row():
                        gamma = gr.Slider(label="gamma", minimum=0.1, maximum=0.9, step=0.05, value=args.gamma)
                    with gr.Row():
                        delta = gr.Slider(label="delta", minimum=0.0, maximum=10.0, step=0.1, value=args.delta)
                    gr.Markdown(f"#### Detector Parameters")
                    with gr.Row():
                        detection_z_threshold = gr.Slider(
                            label="z-score threshold", minimum=0.0, maximum=10.0, step=0.1, value=args.detection_z_threshold
                        )
                    with gr.Row():
                        ignore_repeated_bigrams = gr.Checkbox(label="Ignore Bigram Repeats")
                    with gr.Row():
                        normalizers = gr.CheckboxGroup(
                            label="Normalizations", choices=["unicode", "homoglyphs", "truecase"], value=args.normalizers
                        )
            # with gr.Accordion("Actual submitted parameters:",open=False):
            with gr.Row():
                gr.Markdown(
                    f"_Note: sliders don't always update perfectly. Clicking on the bar or using the number window to the right can help. Window below shows the current settings._"
                )
            with gr.Row():
                current_parameters = gr.Textbox(label="Current Parameters", value=args)
            with gr.Accordion("Legacy Settings", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed_separately = gr.Checkbox(label="Seed both generations separately", value=args.seed_separately)
                    with gr.Column(scale=1):
                        select_green_tokens = gr.Checkbox(label="Select 'greenlist' from partition", value=args.select_green_tokens)

        with gr.Accordion("Understanding the settings", open=False):
            gr.Markdown(
                """
            #### Generation Parameters:

            - Decoding Method : We can generate tokens from the model using either multinomial sampling or we can generate using greedy decoding.
            - Sampling Temperature : If using multinomial sampling we can set the temperature of the sampling distribution. 
                                0.0 is equivalent to greedy decoding, and 1.0 is the maximum amount of variability/entropy in the next token distribution.
                                0.7 strikes a nice balance between faithfulness to the model's estimate of top candidates while adding variety. Does not apply for greedy decoding.
            - Generation Seed : The integer to pass to the torch random number generator before running generation. Makes the multinomial sampling strategy
                                outputs reproducible. Does not apply for greedy decoding.
            - Number of Beams : When using greedy decoding, we can also set the number of beams to > 1 to enable beam search. 
                                This is not implemented/excluded from paper for multinomial sampling but may be added in future.
            - Max Generated Tokens : The `max_new_tokens` parameter passed to the generation method to stop the output at a certain number of new tokens. 
                                    Note that the model is free to generate fewer tokens depending on the prompt. 
                                    Implicitly this sets the maximum number of prompt tokens possible as the model's maximum input length minus `max_new_tokens`,
                                    and inputs will be truncated accordingly.
            
            #### Watermark Parameters:

            - gamma : The fraction of the vocabulary to be partitioned into the greenlist at each generation step. 
                     Smaller gamma values create a stronger watermark by enabling the watermarked model to achieve 
                     a greater differentiation from human/unwatermarked text because it is preferentially sampling 
                     from a smaller green set making those tokens less likely to occur by chance.
            - delta : The amount of positive bias to add to the logits of every token in the greenlist 
                        at each generation step before sampling/choosing the next token. Higher delta values 
                        mean that the greenlist tokens are more heavily preferred by the watermarked model
                        and as the bias becomes very large the watermark transitions from "soft" to "hard". 
                        For a hard watermark, nearly all tokens are green, but this can have a detrimental effect on
                        generation quality, especially when there is not a lot of flexibility in the distribution.

            #### Detector Parameters:
            
            - z-score threshold : the z-score cuttoff for the hypothesis test. Higher thresholds (such as 4.0) make
                                _false positives_ (predicting that human/unwatermarked text is watermarked) very unlikely
                                as a genuine human text with a significant number of tokens will almost never achieve 
                                that high of a z-score. Lower thresholds will capture more _true positives_ as some watermarked
                                texts will contain less green tokens and achive a lower z-score, but still pass the lower bar and 
                                be flagged as "watermarked". However, a lowere threshold will increase the chance that human text 
                                that contains a slightly higher than average number of green tokens is erroneously flagged. 
                                4.0-5.0 offers extremely low false positive rates while still accurately catching most watermarked text.
            - Ignore Bigram Repeats : This alternate detection algorithm only considers the unique bigrams in the text during detection, 
                                    computing the greenlists based on the first in each pair and checking whether the second falls within the list.
                                    This means that `T` is now the unique number of bigrams in the text, which becomes less than the total
                                    number of tokens generated if the text contains a lot of repetition. See the paper for a more detailed discussion.
            - Normalizations : we implement a few basic normaliations to defend against various adversarial perturbations of the
                                text analyzed during detection. Currently we support converting all chracters to unicode, 
                                replacing homoglyphs with a canonical form, and standardizing the capitalization. 
                                See the paper for a detailed discussion of input normalization. 
            """
            )

        gr.HTML(
            """
                <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. 
                    Follow the github link at the top and host the demo on your own GPU hardware to test out larger models.
                <br/>
                <a href="https://huggingface.co/spaces/tomg-group-umd/lm-watermarking?duplicate=true">
                <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                <p/>
                """
        )

        # Register main generation tab click, outputing generations as well as a the encoded+redecoded+potentially truncated prompt and flag
        generate_btn.click(
            fn=generate_partial,
            inputs=[prompt, session_args],
            outputs=[redecoded_input, truncation_warning, output_without_watermark, output_with_watermark, session_args],
        )
        # Show truncated version of prompt if truncation occurred
        redecoded_input.change(
            fn=truncate_prompt, inputs=[redecoded_input, truncation_warning, prompt, session_args], outputs=[prompt, session_args]
        )
        # Call detection when the outputs (of the generate function) are updated
        output_without_watermark.change(
            fn=detect_partial, inputs=[output_without_watermark, session_args], outputs=[without_watermark_detection_result, session_args]
        )
        output_with_watermark.change(
            fn=detect_partial, inputs=[output_with_watermark, session_args], outputs=[with_watermark_detection_result, session_args]
        )
        # Register main detection tab click
        detect_btn.click(fn=detect_partial, inputs=[detection_input, session_args], outputs=[detection_result, session_args])

        # State management logic
        # update callbacks that change the state dict
        def update_sampling_temp(session_state, value):
            session_state.sampling_temp = float(value)
            return session_state

        def update_generation_seed(session_state, value):
            session_state.generation_seed = int(value)
            return session_state

        def update_gamma(session_state, value):
            session_state.gamma = float(value)
            return session_state

        def update_delta(session_state, value):
            session_state.delta = float(value)
            return session_state

        def update_detection_z_threshold(session_state, value):
            session_state.detection_z_threshold = float(value)
            return session_state

        def update_decoding(session_state, value):
            if value == "multinomial":
                session_state.use_sampling = True
            elif value == "greedy":
                session_state.use_sampling = False
            return session_state

        def toggle_sampling_vis(value):
            if value == "multinomial":
                return gr.update(visible=True)
            elif value == "greedy":
                return gr.update(visible=False)

        def toggle_sampling_vis_inv(value):
            if value == "multinomial":
                return gr.update(visible=False)
            elif value == "greedy":
                return gr.update(visible=True)

        def update_n_beams(session_state, value):
            session_state.n_beams = value
            return session_state

        def update_max_new_tokens(session_state, value):
            session_state.max_new_tokens = int(value)
            return session_state

        def update_ignore_repeated_bigrams(session_state, value):
            session_state.ignore_repeated_bigrams = value
            return session_state

        def update_normalizers(session_state, value):
            session_state.normalizers = value
            return session_state

        def update_seed_separately(session_state, value):
            session_state.seed_separately = value
            return session_state

        def update_select_green_tokens(session_state, value):
            session_state.select_green_tokens = value
            return session_state

        # registering callbacks for toggling the visibilty of certain parameters
        decoding.change(toggle_sampling_vis, inputs=[decoding], outputs=[sampling_temp])
        decoding.change(toggle_sampling_vis, inputs=[decoding], outputs=[generation_seed])
        decoding.change(toggle_sampling_vis_inv, inputs=[decoding], outputs=[n_beams])
        # registering all state update callbacks
        decoding.change(update_decoding, inputs=[session_args, decoding], outputs=[session_args])
        sampling_temp.change(update_sampling_temp, inputs=[session_args, sampling_temp], outputs=[session_args])
        generation_seed.change(update_generation_seed, inputs=[session_args, generation_seed], outputs=[session_args])
        n_beams.change(update_n_beams, inputs=[session_args, n_beams], outputs=[session_args])
        max_new_tokens.change(update_max_new_tokens, inputs=[session_args, max_new_tokens], outputs=[session_args])
        gamma.change(update_gamma, inputs=[session_args, gamma], outputs=[session_args])
        delta.change(update_delta, inputs=[session_args, delta], outputs=[session_args])
        detection_z_threshold.change(update_detection_z_threshold, inputs=[session_args, detection_z_threshold], outputs=[session_args])
        ignore_repeated_bigrams.change(
            update_ignore_repeated_bigrams, inputs=[session_args, ignore_repeated_bigrams], outputs=[session_args]
        )
        normalizers.change(update_normalizers, inputs=[session_args, normalizers], outputs=[session_args])
        seed_separately.change(update_seed_separately, inputs=[session_args, seed_separately], outputs=[session_args])
        select_green_tokens.change(update_select_green_tokens, inputs=[session_args, select_green_tokens], outputs=[session_args])
        # register additional callback on button clicks that updates the shown parameters window
        generate_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detect_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        # When the parameters change, display the update and fire detection, since some detection params dont change the model output.
        gamma.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        gamma.change(
            fn=detect_partial, inputs=[output_without_watermark, session_args], outputs=[without_watermark_detection_result, session_args]
        )
        gamma.change(
            fn=detect_partial, inputs=[output_with_watermark, session_args], outputs=[with_watermark_detection_result, session_args]
        )
        gamma.change(fn=detect_partial, inputs=[detection_input, session_args], outputs=[detection_result, session_args])
        detection_z_threshold.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detection_z_threshold.change(
            fn=detect_partial, inputs=[output_without_watermark, session_args], outputs=[without_watermark_detection_result, session_args]
        )
        detection_z_threshold.change(
            fn=detect_partial, inputs=[output_with_watermark, session_args], outputs=[with_watermark_detection_result, session_args]
        )
        detection_z_threshold.change(fn=detect_partial, inputs=[detection_input, session_args], outputs=[detection_result, session_args])
        ignore_repeated_bigrams.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        ignore_repeated_bigrams.change(
            fn=detect_partial, inputs=[output_without_watermark, session_args], outputs=[without_watermark_detection_result, session_args]
        )
        ignore_repeated_bigrams.change(
            fn=detect_partial, inputs=[output_with_watermark, session_args], outputs=[with_watermark_detection_result, session_args]
        )
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[detection_input, session_args], outputs=[detection_result, session_args])
        normalizers.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        normalizers.change(
            fn=detect_partial, inputs=[output_without_watermark, session_args], outputs=[without_watermark_detection_result, session_args]
        )
        normalizers.change(
            fn=detect_partial, inputs=[output_with_watermark, session_args], outputs=[with_watermark_detection_result, session_args]
        )
        normalizers.change(fn=detect_partial, inputs=[detection_input, session_args], outputs=[detection_result, session_args])
        select_green_tokens.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        select_green_tokens.change(
            fn=detect_partial, inputs=[output_without_watermark, session_args], outputs=[without_watermark_detection_result, session_args]
        )
        select_green_tokens.change(
            fn=detect_partial, inputs=[output_with_watermark, session_args], outputs=[with_watermark_detection_result, session_args]
        )
        select_green_tokens.change(fn=detect_partial, inputs=[detection_input, session_args], outputs=[detection_result, session_args])

    demo.queue()

    if args.demo_public:
        demo.launch(share=True)  # exposes app to the internet via randomly generated link
    else:
        demo.launch()


def getNoisyData(no_wm_text, wm_text, num_instances, ratio):
    noise_ratio = math.floor(len(wm_text) * ratio / 100)
    noise_indices = np.random.randint(0, len(wm_text), num_instances)
    noise_snippets = [wm_text[index : index + noise_ratio] for index in noise_indices]

    text_indices = np.random.randint(0, len(no_wm_text), num_instances)
    return_text = no_wm_text
    for key, value in enumerate(text_indices):
        return_text = return_text[:value] + noise_snippets[key] + return_text[value + noise_ratio :]

    return return_text


def get_roc_auc_scores(z_scores_no_wm, z_scores_wm):
    z_scores = np.concatenate([np.array(z_scores_no_wm), np.array(z_scores_wm)])
    class_labels = np.concatenate([np.zeros_like(z_scores_no_wm), np.ones_like(z_scores_wm)])

    result = metrics.roc_curve(class_labels, z_scores, pos_label=1)
    return metrics.auc(result[0], result[1])


def llm_paraphrasing(prompt, text):
    response = model.generate_content(prompt + text)
    if response:
        return response.text


def read_jsonl(file_path, columns):
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            selected_data = {col: line[col] for col in columns if col in line}
            data.append(selected_data)
    return data


def main(args):
    """Run a command line version of the generation and detection operations
    and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = args.normalizers.split(",") if args.normalizers else []

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    # Generate and detect, report to stdout
    if not args.skip_model_load:
        if args.elifive:
            data = read_jsonl("watermark_reliability_release/utils/data/lfqa.jsonl", ["title", "selftext"])
            data = np.random.choice(data, 2)

            formatted_data = [{"text": item["title"] + " Premise: " + item["selftext"]} for item in data]

            perplexity = load("perplexity", module_type="metric")

            for item in tqdm(formatted_data):
                input_text = "Answer the following question in 200-300 words. Explain it like I'm five.\n\n" + item["text"]
                args.default_prompt = input_text
                _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(
                    input_text, args, model=model, device=device, tokenizer=tokenizer
                )
                without_watermark_detection_result = detect(decoded_output_without_watermark, args, device=device, tokenizer=tokenizer)
                with_watermark_detection_result = detect(decoded_output_with_watermark, args, device=device, tokenizer=tokenizer)

                item["wm_text"] = decoded_output_with_watermark
                item["len_wm_text"] = len(decoded_output_with_watermark)

                for sublist in without_watermark_detection_result[0]:
                    if sublist[0] == "z-score":
                        item["without_wm_scores"] = float(sublist[1])
                        break

                item["no_wm_text"] = decoded_output_without_watermark
                item["len_no_wm_text"] = len(decoded_output_without_watermark)

                for sublist in with_watermark_detection_result[0]:
                    if sublist[0] == "z-score":
                        item["with_wm_scores"] = float(sublist[1])
                        break

                try:
                    results = perplexity.compute(
                        model_id="facebook/opt-1.3b",
                        add_start_token=False,
                        predictions=[decoded_output_with_watermark, decoded_output_without_watermark],
                    )

                    item["ppl_score_wm"] = round(results["perplexities"][0], 2)
                    item["ppl_score_no_wm"] = round(results["perplexities"][1], 2)
                except:
                    continue

            with open("result_eli5.json", "w") as fp:
                json.dump(formatted_data, fp)

        if args.cp_attack:
            # news = [
            #     {
            #         "text": "What made you a music director? As a Bengali with an MBBS degree, did you never feel the pressure to pursue a medical career?\nThere is always pressure on you while taking risks. Opposition from family and society is natural. I guess that���s how the society reacts. But some of us are just not made to follow the societal norms. My immense love for music coupled with my adamant nature didn���t let me settle for anything apart from music. For me, music is like medicine.\nDespite hailing from Kolkata, you never thought of composing for Bengali films.\nOf course I do. I���d love to work with Srijit Mukherji and Kaushik Ganguly.\nHow much are you clued in to Bengali film music?\nI left Kolkata around eight years back and Bengali film music has come a long way since then. I love Anupamda���s (Roy) songwriting and compositions. He is brilliant. The song Ek baar bol tor keu nei is one of my favourites.\nHeard you used to fight with your mother as a kid because you were forced to learn Rabindrasangeet?\nOh yes! Protidin ma er sanghe jhogra hotoh gaan sekha niye! But now Rabindrasangeet is my fave. There are times when I just sit quietly and listen to the songs. During my college days, I used to love rock. Fossils was at its peak then. I grew up listening to Kabir Suman, Anjan Dutt and so on.\nDo you catch up with your old friends whenever you come down to Kolkata?\nI do meet my school friends when I come here. Right now, Kolkata seems to be brimming with joy as Durga Puja is just weeks away. But sadly, I���m going to miss it this time as I would be out of the country for work. What I will miss the most are pandal-hopping and eating out in the wee hours.\nAre you in touch with other Bengali music directors based in Mumbai?\nOf course. I have known Pritamda very well since I was nominated for my debut in Jism2 and for my second film, Yaariyan. I was lucky to score the soundtrack along with him, Mithoon and YoYo Honey Singh. We won the best music award for it too. Pritamda is very warm and often invites us over to his house. Jeetda (Gannguli) is like family to me. His wife is my mom���s student and they are the sweetest people on earth. We often hang out in a group for adda where we don���t talk work. I don���t know Shantanu Moitra personally, but whenever we meet he is extremely encouraging and supportive.\nHow is the competition in Mumbai music industry?\nQuite tough, but fair at the same time. What matters there is your song���s merit and nothing else.\nMost of your songs, which are also penned by you, are hits. Does that create a pressure on you when you compose your next song?\nThere are certain expectations when you work with big stars and banners. But I take it more as a compliment than a burden.\nWell the list is endless -- my childhood, daily experiences, the people around, my lovers, travelling and imagination.\nWhile promoting Rustom in Kolkata, Akshay Kumar crooned Tere sang yaara, which is composed by you. How does it feel?\nFeels great. He told me that after a long time he has a favourite song. Akshay sir has always been very kind to me. This is the third song I made for him after Meherbaani from The Shaukeens and the title track for a TV show hosted by him called Dare 2 Dance.\nWho are your favourite male and female singers?\nAmong male artistes, it���s Chris Cornell, Eddie Vedder, Bruce Springsteen, Ustad Nusrat Fateh Ali Khan, Hemant Mukherjee and Mohammed Rafi to name a few. Among female singers, Lauryn Hill, Adele, Alanis Morissette, Abida Parveen and Shreya Ghoshal are my favourites.\nWhich music director do you like the most?\nMy all-time favourite music director is Hemanta Mukherjee. I am also very fond of Vishal Bhardwaj.\nOne singer you wish to lend voice to your music?\nMy dream is to have Ustad Nusrat Fateh Ali Khan or Jagjit Singh lend their voice to music composed by me some day.\nWhich is your favourite composition till date?\nDariya, from an upcoming film starring Katrina Kaif and Sidharth Malhotra. The song conveys the simplest and purest emotion and that���s how music should be.\nAkshay Kumar, Sidharth Malhotra, Irrfan, Randeep Hooda all have lip-synced on your tracks. Any actor you would want to playback for?\nI would like to playback for Ranbir Kapoor. He is one of my favourite actors."
            #     },
            #     {
            #         "text": "He was in the airline business. He was American. And he knew his golf.\nYou know, he said, the American golfer who makes a first foray across the Atlantic typically goes to Scotland first.\nThen the golfer goes to Ireland.\nThis, of course, begged the question: why?\nI’ll attempt to answer, but first an admission.\nI am biased because I am from Ireland and much of my youth was spent of golf courses there, one in particular, where my dad and other family members enjoyed membership.\nThat said, I am also objective enough to know that in writing about Ireland as one of the planet’s prime golf destinations I am on very firm ground indeed, anything but a voice in the wilderness, or the rough.\nAnd here’s another thing. I have played the game all over the United States and that includes Hawaii.\nI have walked the walk at the likes of Augusta National and Pinehurst. I have cast an admiring eye over Pebble Beach.\nAnd yet I still close my eyes and imagine a fair Irish day on a green Irish fairway.\nI also imagine better shots than I’m capable of hitting, but that’s another story.\nAt the end of this homage to Irish golf - being penned just days before the U.S. Open in a place called Erin Hills (Wisconsin) no less - I will repeat what has been said to me by more than one American golfer, this with regard to where arguably the most spectacular golf course on planet earth can be found.\nIt’s an Irish course…of course.\nAnd, yes, this is a subjective point of view. It’s really impossible to rate a course as being “best” in the world. But “spectacular” leaves a little more wiggle room.\nAnd, as stated, the affirming with regard to this particular course opinion has been directed at yours truly more than once.\nI have not played this course, but have played a fair few other Irish courses that certainly rate in global terms.\nSo let’s take a little trip around the island and name a few.\nFirst up though, it’s important to note that golf, along with a number of other sports, is an all island affair in terms of governance.\nThere is no border in Irish golf.\nThis fact led to all the ballyhoo a while back regarding Rory McIlroy and whether he would represent Ireland or Great Britain in the Olympics.\nHe didn’t play in the games, but had he done so it would have been for Ireland.\nRory is a pivotal figure in Irish golf in more ways than one.\nHis foundation is a prime sponsor of the Irish Open, this year being played in Portstewart, County Derry in early July.\nMcIlroy’s success in golf majors, along with that of Graeme McDowell and Darren Clarke, both from Northern Ireland, has been instrumental in the decision to bring The Open to Royal Portrush, just down the road from Portstewart, in 2019.\nPadraig Harrington’s name being twice inscribed on the Claret Jug is also a factor.\nThis return of The Open to the Antrim coast draws a link back to 1951 when the oldest of golf’s majors was played there and won by Max Faulkner, a player remembered for his preference for plus fours instead of golfing slacks.\nFaulkner was a standout in more ways than just dress.\nThe story has it that after three rounds through the rolling Portrush dunes, Faulkner was signing autographs that already proclaimed himself as the winner of the tournament.\nTrue, he was six shots up after 54 holes but stranger things have happened. In this case they didn’t.\nI met Max Faulkner when I was a kid.\nHe was playing at that course where I spent so much of my youth, Woodbrook, which straddles the line between Dublin and Wicklow.\nI remember those odd looking plus fours. Faulkner was also a nice guy, or at least a tolerant one as he was being buzzed by whippersnappers like myself eager for player monikers.\nRoyal Portrush and Portstewart rate highly then.\nAnd in Ulster there’s another brace to add to the list of must-plays. Royal County Down for sure, and I have fond memories of a nice stroll around Rosapenna links in County Donegal, a place where you could tee off on the eighteenth at 10.30 on a midsummer’s night and, assuming no diversions into the rough, finish in a manageable gloaming.\nBeing an island, Ireland has an abundance of links courses but also some truly beautiful parkland venues.\nThe K Club in Kildare and Mount Juliet in Thomastown, Co. Kilkenny come to mind.\nIn the case of the latter, there is a claim that this is the best parkland course on the island. Well, it was designed by Jack Nicklaus so that’s an argument that would be difficult to refute.\nDruid’s Glen in County Wicklow is definitely worth a visit. Killarney, which I have battled my way around, is for sure another as is Adare Manor in County Limerick.\nThe aforementioned Woodbrook, which hosted a number of Ireland’s top tournaments in the 1960s and 1970s, is beside the Irish Sea but is rated a parkland course because it is fronted by mud cliffs.\nThe wind has had its wicked way with some of the course trees, not least pines that end up leaning this way and that.\nSometimes, with a quick glance at certain corners of the course, you would actually be reminded a little of Pebble Beach.\nIt is probably fair to say that the American golfer, when thinking in terms of an Irish foray, first and foremost considers a links.\nNo shortage of those of course. From those northern bastions mentioned above come on down the coast stopping at Rosses Point in County Sligo, Lahinch in Clare and also in the Banner County, Doonbeg and its mountainous dunes.\nDoonbeg has Greg Norman’s name on the designer credits. He is reputed to have turned up, cast an eye over the landscape only to proclaim that there was nothing he could do here that God had not already done.\nHaving played Doonbeg, I would confidently second that assertion.\nHaving played it in a fresh Atlantic wind I would throw in the devil himself for good measure.\nDown the coast a bit there is mighty, and mighty famous, Ballybunion with that opening tee shot that invites you to land your ball in a cemetery.\nA fitting metaphor for my game so when I stood on the tee I made sure that I aimed well left of the headstones.\nWaterville in County Kerry is another course to tackle. This was the late great Payne Stewart’s home from home. He used to tend bar in the clubhouse and was proclaimed honorary captain.\nThere is a statue of Stewart at Waterville that is the Irish match for the bronze that can be viewed at Pinehurst in North Carolina.\nThe Waterville website mentions that less than one percent of the world’s golf courses rate as true links courses, and that 85 percent of them can be found in Ireland and Britain.\nWaterville asserts that it is the greatest links in the Republic, while granting the title to Royal County Down in Northern Ireland.\nThere would be argument over that of course, but both courses are for sure prominent in said argument.\nCloser to Dublin there are a fair few fine links tracks, notably Royal Dublin and Portmarnock, while down the coast a bit, in County Wicklow, there is a true gem called the European Club.\nBut what of that as yet unnamed course so beloved by those peripatetic American eyes?\nMany years ago, the great journalist and broadcaster, Alistair Cooke, spun out a tease on television.\nAt the end of the broadcast he promised that he would deliver his verdict on the woman that he believed was the most beautiful in living memory.\nWell, he went this way and that until the very end at which point he proclaimed his belief that the actress Ava Gardner was his one and only.\nSuch a proclamation was certain to inspire debate of course.\nEvery Hollywood star has a fan club. Every great golf course has a fan club.\nAnd Irish golfers will spend many a long evening arguing for this course and that course – and never quite reach a conclusion.\nBut this assertion is based on the verdict, arguably more objective, of a slew of American golfers who have played this course, or have simply clapped eyes on it, and who have come within earshot of yours truly while announcing their verdict.\nSo this jury is in.\nTo describe it as the “best” course is not applicable because that invites subjective argument from here to eternity.\nBut to describe this course as being the most stunning sight in the most spectacular setting invites greater consensus.\nSo without further ado….Well, with just a little because the reader can have a moment to guess.\nThe course in question can be found on a clenched fist of land off the Cork coast.\nI am speaking (of course!) of Old Head.\nThis promontory peach has prompted a unanimity to a degree and pitch that I have not heard applied to any other golf course in Ireland, and from some, the world.\nReaders might beg to differ, but best to do that after playing Old Head which is just a few miles from Kinsale.\nIn a way, Old Head is itself a metaphor for the entire island when it comes to golf.\nIt reaches out into the sea from a much larger landmass, just as Ireland itself sits in the sea off a much larger landmass.\nBoth metaphor and reality share their other worldly and spectacular natures.\nAnd that’s beyond any argument in any club house bar.\nWe will be reminded of this very soon with the Irish Open at Portstewart, and again two years from now when The Open returns to Royal Portrush.\nFinally, back to that question at the top: Scotland, Ireland, Ireland again. Why?\nNow that’s a tough one but I suspect that the “craic,” as they say, might just be a little mightier in Hibernia than Caledonia.\nOne to argue over a wee dram ... of Irish, of course."
            #     },
            #     {
            #         "text": "For professors, publishing in elite journals is an unavoidable part of university life. The grueling process of subjecting work to the up-or-down judgment of credentialed scholarly peers has been a cornerstone of academic culture since at least the mid-20th century.\nNow some humanities scholars have begun to challenge the monopoly that peer review has on admission to career-making journals and, as a consequence, to the charmed circle of tenured academe. They argue that in an era of digital media there is a better way to assess the quality of work. Instead of relying on a few experts selected by leading publications, they advocate using the Internet to expose scholarly thinking to the swift collective judgment of a much broader interested audience.\nThat transformation was behind the recent decision by the prestigious 60-year-old Shakespeare Quarterly to embark on an uncharacteristic experiment in the forthcoming fall issue — one that will make it, Ms. Rowe says, the first traditional humanities journal to open its reviewing to the World Wide Web.\nMixing traditional and new methods, the journal posted online four essays not yet accepted for publication, and a core group of experts — what Ms. Rowe called “our crowd sourcing” — were invited to post their signed comments on the Web site MediaCommons, a scholarly digital network. Others could add their thoughts as well, after registering with their own names. In the end 41 people made more than 350 comments, many of which elicited responses from the authors. The revised essays were then reviewed by the quarterly’s editors, who made the final decision to include them in the printed journal, due out Sept. 17.\nThe Shakespeare Quarterly trial, along with a handful of other trailblazing digital experiments, goes to the very nature of the scholarly enterprise. Traditional peer review has shaped the way new research has been screened for quality and then how it is communicated; it has defined the border between the public and an exclusive group of specialized experts.\nToday a small vanguard of digitally adept scholars is rethinking how knowledge is understood and judged by inviting online readers to comment on books in progress, compiling journals from blog posts and sometimes successfully petitioning their universities to grant promotions and tenure on the basis of non-peer-reviewed projects.\nThe quarterly’s experiment has so far inspired at least one other journal — Postmedieval — to plan a similar trial for next year.\nJust a few years ago these sorts of developments would have been unthinkable, said Dan Cohen, director of the Center for History and New Media at George Mason University. “Serious scholars are asking whether the institutions of the academy — as they have existed for decades, even centuries — aren’t becoming obsolete,” he said.\nEach type of review has benefits and drawbacks.\nThe traditional method, in which independent experts evaluate a submission, often under a veil of anonymity, can take months, even years.\nClubby exclusiveness, sloppy editing and fraud have all marred peer review on occasion. Anonymity can help prevent personal bias, but it can also make reviewers less accountable; exclusiveness can help ensure quality control but can also narrow the range of feedback and participants. Open review more closely resembles Wikipedia behind the scenes, where anyone with an interest can post a comment. This open-door policy has made Wikipedia, on balance, a crucial reference resource.\nMs. Rowe said the goal is not necessarily to replace peer review but to use other, more open methods as well.\nIn some respects scientists and economists who have created online repositories for unpublished working papers, like repec.org, have more quickly adapted to digital life. Just this month, mathematicians used blogs and wikis to evaluate a supposed mathematical proof in the space of a week — the scholarly equivalent of warp speed.\nIn the humanities, in which the monograph has been king, there is more inertia. “We have never done it that way before,” should be academia’s motto, said Kathleen Fitzpatrick, a professor of media studies at Pomona College.\nMs. Fitzpatrick was a founder of the MediaCommons network in 2007. She posted chapters of her own book “Planned Obsolescence” on the site, and she used the comments readers provided to revise the manuscript for NYU Press. She also included the project in the package she presented to the committee that promoted her to full professor this year.\nMany professors, of course, are wary of turning peer review into an “American Idol”-like competition. They question whether people would be as frank in public, and they worry that comments would be short and episodic, rather than comprehensive and conceptual, and that know-nothings would predominate.\nAfter all, the development of peer review was an outgrowth of the professionalization of disciplines from mathematics to history — a way of keeping eager but uninformed amateurs out.\n“Knowledge is not democratic,” said Michèle Lamont, a Harvard sociologist who analyzes peer review in her 2009 book, “How Professors Think: Inside the Curious World of Academic Judgment.” Evaluating originality and intellectual significance, she said, can be done only by those who are expert in a field.\nThe most daunting obstacle to opening up the process is that peer-review publishing is the path to a job and tenure, and no would-be professor wants to be the academic canary in the coal mine.\nAlthough initially cautious, Mr. Galey said he is now “entirely won over by the open peer review model.” The comments were more extensive and more insightful, he said, than he otherwise would have received on his essay, which discusses Shakespeare in the context of information theory.\nAdvocates of more open reviewing, like Mr. Cohen at George Mason argue that other important scholarly values besides quality control — for example, generating discussion, improving works in progress and sharing information rapidly — are given short shrift under the current system.\n“There is an ethical imperative to share information,” said Mr. Cohen, who regularly posts his work online, where he said thousands read it. Engaging people in different disciplines and from outside academia has made his scholarship better, he said."
            #     },
            #     {
            #         "text": "The NHS is still running Windows XP en masse, two and a half years after Microsoft stopped delivering bug fixes and security updates.\nNearly all of England NHS trusts – 90 per cent – continue to rely on PCs installed with Microsoft’s 15-year-old desktop operating system.\nJust over half are still unsure as to when they will move to a replacement operating system.\nFourteen per cent reckoned they’d move to a new operating system by the end of this year, and 29 per cent reckoned the move would happen “some time” in 2017.\nWindows XP is not receiving any security updates from Microsoft, meaning health service PCs are wide open to hackers and malware.\nThe data on the NHS' use of Windows XP comes courtesy of a Freedom of Information request from Citrix, which approached 63 NHS trusts and received responses from 42.\nAn FoI request from Citrix made in July 2014, three months after Microsoft’s deadline to move off Windows XP, had found 100 per cent of NHS trusts were dependent on the operating system.\nThe Reg first reported in early 2014 how vast sections of the UK public sector was set to miss Microsoft’s April 2014 kill date for XP.\nThe government had agreed a temporary framework support agreement with Microsoft which guaranteed delivery of special security patches for a year.\nThat agreement ended on April 14 2015 after it was decided not to go for a second year."
            #     },
            #     {
            #         "text": "These are commercial building permits on file with Cleveland County for the month of September.\nLocation: 525 W. Zion Church Rd.\nLocation: 110 S. Main St.\nLocation: 4357 W. Dixon Blvd."
            #     },
            #     {
            #         "text": "the World Challenge a year ago. He won at Hilton Head on the U.S.\nChampionships in Doral and Shanghai.\nthe way up to No. 12.\nWoods won the vote as the best player on the U.S. PGA Tour.\nand Firestone – and there would be no debate.\nwho are all on form.\nmuch weight is given a major.\nthat not even he thought he could win.\nWho won the most meaningful major this year? Mickelson or Scott?\nBest to save that argument for the bar.\nsummer when the Swede began to shine.\nCup and Race to Dubai in the same season.\nyear at No. 1 – McIlroy. He still had a good view.\nto win. Henrik comes back,” McIlroy said. ”Yeah, it’s deep.\nuntil she faltered in the Titleholders."
            #     },
            #     {
            #         "text": "Under intense pressure to improve conditions in the jail complex on Rikers Island, the administration of Mayor Bill de Blasio has developed a plan to move 16- and 17-year-olds to a dedicated jail for youths in the Bronx.\nThe cost to carry out the plan is expected to be about $300 million, officials said.\nThe plan calls for the city to reconfigure the Horizon Juvenile Center, which is currently used to hold 14- and 15-year olds, to house the 16- and 17-year olds who are typically sent to Rikers.\nA 2015 settlement with the Department of Justice on reform at Rikers called on the city to seek an alternative location to house inmates under 18, although it stopped short of requiring it.\nNew York is the only state other than North Carolina that prosecutes all children 16 and older as adults if they are accused of a crime. In New York City, they are likely to wind up at Rikers, a notoriously brutal lockup. There are currently about 200 inmates who are 16 or 17 at Rikers, down from about 330 in 2013.\nThe city and many advocates have urged state lawmakers to pass a law that would treat those under the age of 18 as juveniles, but the proposal has gone nowhere in Albany.\nThe new plan to move 16- and 17-year-olds from Rikers must overcome several hurdles. It has to be approved through the city’s time-consuming land use process: The local community board and the Bronx borough president get to weigh in and it must be approved by the City Planning Commission and the City Council.\nBut the change will not happen any time soon; officials said it could take four years or more to get approvals and to complete construction. The plan also calls for the city to remodel another juvenile detention site, the Crossroads Juvenile Center in Bushwick, Brooklyn, to hold all of the city’s 14- and 15-year-old detainees, including those who had previously gone to Horizon.\nThe cost of refurbishing the Bronx center is budgeted at $170 million. The cost of remodeling the Brooklyn center is budgeted at $129 million.\nAdvocates supported the move but lamented the long delay before the new center will be ready, assuming that it survives the land-use review process.\n“For us this is a marathon, not a sprint.” he said.\nMore than 95 percent of the 16- and 17-year-olds at Rikers are awaiting trial. More than a third have been charged with robbery and about one in 10 have been charged with assault, according to data provided by the city.\nIncreased attention was focused on the plight of younger teenagers at Rikers in 2014 after The New Yorker published an article about Kalief Browder, who was sent there at 16, accused of stealing a backpack. He never stood trial or was found guilty of any crime but he spent three years at Rikers, nearly two of them in solitary confinement. He told of being beaten repeatedly by guards and other inmates and trying several times to kill himself while in custody. After his release he remained deeply troubled by the experience and he committed suicide last year at age 22.\nThe city ended solitary confinement for Rikers inmates under 18 in December 2014."
            #     },
            #     {
            #         "text": "WASHINGTON -- With television lights glaring, 20 lawmakers will gather next week to revisit the fight that consumed Congress before Christmas over renewing a Social Security payroll tax cut and unemployment benefits.\nLittle real work will be done, but the meeting will mark the formal start of an effort to untangle a dispute that both parties want to resolve, though for different reasons. Following is a look at the path Round 2 could take, based on interviews with participants on both sides.\nQ: Can you remind me what's at stake?\nA: After a bitter clash and just a week before a New Year's Day deadline, President Barack Obama and Congress renewed a 2 percentage point payroll tax cut for 160 million workers and benefits for the long-term unemployed through February. They also temporarily forestalled a deep cut in doctors' Medicare fees that threatened to make it harder for the elderly to find physicians who would treat them. Now, the two sides need to figure out how to extend all three measures through 2012 and cover the roughly $160 billion cost.\nQ: Are they expected to succeed?\nA: Yes, though it will probably take until shortly before the current extensions expire Feb. 29. There are complicated decisions ahead, chiefly what programs to cut and what fees to increase to offset the price tag. Just as important, Democrats won't be in a hurry to finish.\nA: Republicans took a severe pounding in December when the House GOP resisted a bipartisan, Senate-approved, two-month extension of the payroll tax cut, which was designed to give lawmakers time to negotiate a longer version. With control of the White House and Congress at stake in the November elections, many Democrats think the GOP could incur further damage if these latest talks take time.\nMany Republicans doubt the economic benefit of a payroll tax cut, a foundation of Obama's plan to create jobs. But as December's battle unfolded, GOP leaders worried that they would suffer political damage from opposing the deeply popular tax cut, worth $1,000 annually to a family earning $50,000 a year.\nWith the House's fractious conservative wing balking until the very end, the fight made the GOP look like it was opposing the tax reduction -- which Democrats contrasted with Republican support for tax breaks for the wealthy. Most Republicans want this year's fight to end quickly so they can change the subject to their own efforts to cut taxes, federal spending and Obama administration regulations.\nQ: How long can Democrats prolong the negotiations?\nA: If they're not careful they could overplay their hand.\nDemocrats scored points last year by forcing Senate votes on their proposal to finance the payroll tax cut with a small surtax on people earning $1 million or more a year. They have a new incentive to do something similar this year with the GOP presidential front-runner Mitt Romney, a wealthy venture capitalist, being cast by party rivals as callous and out of touch.\nAs a result, many Democrats want to begin this year's talks on extending the Social Security tax cut by targeting the wealthy for a tax increase, perhaps with the millionaire surtax or by limiting their deductions. The millionaire surtax has no chance of passage in the GOP-run House, and Democrats could be accused of blatantly playing politics. Democrats and Obama have a reason to cut a deal: They believe extending the payroll tax cut and jobless benefits will goose the economy and reduce the risk of another economic downturn that could hurt their election prospects.\nQ; What will the 20 members of Congress do?\nA: House and Senate party leaders each have appointed bargainers to hash out differences over the bill, following Congress' tradition of naming conference committees to craft compromise legislation. But as usual when high-profile battles are being resolved, party leaders will have tight control over the ultimate deal. Still, conference committee members will play a role in writing details, and their endorsement of a package would let leaders argue that they didn't jam something down the throats of rank-and-file lawmakers.\nQ: Who are these 20 lawmakers?\nA: They range from formidable committee chairmen to lowly freshmen, but each has a stake in the fight.\nThe chairmen of Congress' two tax-writing committees are included: Rep. Dave Camp, R-Mich., of the House Ways and Means Committee, and Sen. Max Baucus, D-Mont., of the Senate Finance Committee.\nSen. Jon Kyl, R-Ariz., is the Senate's No. 2 Republican and a close ally of Senate Minority Leader Mitch McConnell, R-Ky. Democratic Sen. Bob Casey, facing re-election this fall in the pivotal state of Pennsylvania, has repeatedly been given a visible role in the payroll tax fight by party leaders.\nRep. Xavier Becerra, D-Calif., a party leader, should be a leading opponent of Republican proposals to help finance the plan by effectively denying the child tax credit to many illegal immigrants. Freshman GOP Rep. Nan Hayworth is from a closely contested district in New York's Hudson River Valley.\nHayworth and Rep. Tom Price, R-Ga., are doctors, which could give them roles in the talks involving Medicare. A pair of Maryland Democrats, Sen. Ben Cardin and Rep. Chris Van Hollen, are sure to battle a Republican proposal to make federal employees contribute more to their pensions.\nQ: Do they bring other experience to the bargaining table?\nA: Seven have participated in recent, failed bipartisan efforts to contain mammoth budget deficits. Those were Congress' supercommittee, talks led by Vice President Joe Biden, the \"Group of Six\" senators, and a presidential commission headed by former Wyoming GOP Sen. Alan Simpson and former President Bill Clinton's White House chief of staff, Erskine Bowles.\nNone of those groups succeeded, largely because party leaders could not agree to the controversial tax increases and cuts in entitlement programs like Medicare that would have been required for the trillions of dollars in savings needed.\nFar smaller savings are needed to resolve the payroll tax fight, and the consensus is that this time, the president and leaders in both parties want a package that can become law."
            #     },
            #     {
            #         "text": "AUSTIN, Texas — Awkward kisses, emoji and Topanga. That's what love is all about for the cast of Undateable.\nChris D’Elia, Brent Morin, Ron Funches, Rick Glassman, Bridgit Mendler, Bianca Kajlich, David Fynn took a pause from the craziness of SXSW for a little romance. Or at least to talk about romance. And to kiss.\nUndateable airs Tuesdays on NBC."
            #     },
            #     {
            #         "text": "WASHINGTON - WASHINGTON (AP) — Federal regulators voted Wednesday to require companies to reveal more information about how they pay their executives amid a public outcry over compensation.\nThe Securities and Exchange Commission voted 4-to-1 to expand the disclosure requirements for public companies.\nCompany policies that encouraged excessive risk-taking and rewarded executives for delivering short-term profits were blamed for fueling the financial crisis.\nThe SEC also changed a formula that critics say allowed companies to understate how much their senior executives are paid. At issue is how public companies report stock options and stock awards in regulatory filings. Such awards often make up most of top executives' pay.\nThe new requirements include information on how a company's pay policies might encourage too much risk-taking.\nSeparately, the agency voted unanimously to require thousands of investment advisers who have custody of clients' money to submit to annual surprise exams by outside auditors.\nThe surprise audits would allow independent accountants to review the books and verify that the money is there. The snap audits would apply to about 1,600 investment advisers that don't use third-party custodians, out of roughly 11,000 advisers registered with the SEC.\nThis move is aimed at plugging gaps that allowed disgraced money manager Bernard Madoff to deceive investors.\nThe expanded executive pay disclosure rules will take effect next spring, when companies send annual proxy disclosures to shareholders.\nThe changes will help investors make better-informed voting decisions for the companies in which they hold stock, SEC Chairman Mary Schapiro said.\n\"By adopting these rules, we will improve the disclosure around risk, compensation and corporate governance, thereby increasing accountability and directly benefiting investors,\" Schapiro said before the vote.\nBut Commissioner Kathleen Casey said she opposed some of the new requirements, such as added information on qualifications of directors and candidates for the board, that she said could be \"unduly burdensome.\"\nAs a result, Casey said she was voting against the rule as a whole.\nIt was the first final rule adopted by the SEC this year under Schapiro's tenure. Numerous proposals have been made by the commissioners.\n—Legal actions involving the company's executive officers, directors and nominees for the board.\n—The role played by diversity as a factor in choosing candidates for the board.\n—Potential conflicts of interest on the part of compensation consultants retained by the company.\nThe Obama administration imposed pay curbs on banks that received federal bailout money. Since then, eight of the largest such banks have either repaid or said they will repay their federal money, largely to escape caps on executive pay.\nThe Federal Reserve has set a February deadline for the 28 biggest U.S. banks — including Goldman Sachs, JPMorgan Chase & Co., Citigroup Inc., Bank of America Corp. and Wells Fargo & Co. — for submitting 2010 compensation plans. The Fed also will be encouraging, though not requiring, banks to revise this year's pay plans if they are out of step with principles the Fed has proposed to limit risk.\nAnger over lavish Wall Street pay has led some U.S. banks to take pre-emptive action. Goldman Sachs, for example, has said it won't give cash bonuses to 30 top executives. Instead, the bonuses will be paid in stock that can't be cashed in for five years.\nCompanies will have to disclose how pay is determined in departments involved in the riskiest activities — or departments that produce a big chunk of company profits.\nThe new requirements were proposed by the SEC and opened to public comment in July. They build on rules the agency adopted in 2006.\nUnder current rules, companies don't have to reveal the full value of stock options they give an executive. Instead, they must disclose in their annual proxy statements only the portion of an options award that vests that year.\nThe new rule will require companies to show in a summary table the estimated value of all stock-based awards on the day they are granted. The SEC's 2006 rules had relegated those totals to a separate table that investors often overlook or find hard to decipher.\nAn example is the case of a company that decides its CEO deserves $10 million worth of stock options, to vest in equal installments over four years. Under current rules, the company would have to include only $2.5 million — one-fourth of the total of the $10 million total — in the summary table."
            #     },
            #     {
            #         "text": "Noble Prize winner Baruch Blumberg will present the 2009 Saxon Graham lecture on April 16.\nBUFFALO, N.Y. -- Baruch S. Blumberg, M.D., Ph.D., winner of the 1976 Nobel Prize in Medicine for discovery of the hepatitis B virus (HBV), will present a talk on \"The Adventure of Science and Discovery,\" April 16 at 5 p.m. in Butler Auditorium in Farber Hall on the University at Buffalo's South (Main Street) Campus.\nBlumberg has had a major impact on worldwide public health throughout his career. He and his colleagues were responsible for developing the HBV vaccine, which has decreased HBV infection dramatically along with the incidence of liver cancer that can be caused by HBV.\nThe virus is an important cause of disease and death in many populous nations, especially Asia and Africa. The vaccine and the diagnostic tests that followed the discovery of the virus have saved millions of lives.\nBlumberg is a professor of medicine and anthropology at the University of Pennsylvania and Distinguished Scientist at the Fox Chase Cancer Center in Philadelphia. More recently, he has been involved in research at the National Aeronautics and Space Administration, where he is director of the NASA Astrobiology Institute. The institute concentrates on studying the origin, evolution, distribution and future of life in the universe.\nBlumberg will discuss both his work with HBV and his work on astrobiology during his lecture.\nAmong his many affiliations, Blumberg is a member of the National Academy of Sciences, the American Philosophical Society and the National Inventors Hall of Fame.\nSponsored by the UB Department of Social and Preventive Medicine (SPM), the lecture is part of the Saxon Graham Lectureship series.\nAn accomplished epidemiologist, Graham chaired the Department of Social and Preventive Medicine from 1981 to 1991. He is known for his important contributions to the understanding of the impact of diet on cancer, many of which were based on studying dietary habits of Western New Yorkers.\nThe department, an integral component of UB's School of Public Health and Health Professions, continues Graham's legacy of using epidemiologic tools in research studies to understand to the causes and prevention of diseases in human populations, especially in the Western New York community."
            #     },
            #     {
            #         "text": 'The Mapes family of Effingham enjoy the Lincoln Park Zoo in Chicago with their children including their adopted children, Regino and Regina, who were born in the Philippines.\nMisty Mapes and her husband, Patrick, of Effingham always had a desire to add to their family through adoption.\nThat dream became a reality in part due to Gift of Adoption Fund, a nonprofit organization that provides financial support to families that need help to pay for the hefty cost of adopting a child.\nThe Mapes, who have two biological children, Braydon, 16, and Madison, 11, were able to adopt 8-year-old twins, Regina and Regino — who go by Ina and Ino — about a year ago from the Philippines. They received a $4,000 grant that helped them pay travel expenses to the Philippines to bring the children home.\nThough they had always had tossed around the idea of adoption, they were spurred to take action when their older son asked them about it a few years ago.\n"He said, \'Hey. Can we adopt? I\'d like to have a brother," Misty Mapes recalled.\nMisty Mapes, who works as a teacher, and Patrick, who is a dockworker, set about eliminating as many of their expenses as they could to save the $40,000 they would need to took to fund the adoption.\nThat final boost was provided by the Gift of Adoption Fund, which bridges the game when people are nearing the goal of completing the adoption, but need a little extra financial help to finalize it, said Marcy McKay, a La Grange resident and volunteer for the fund.\nThough they didn\'t need the financial help to adopt their children that the Gift of Adoption Fund provides, the McKays, who have three adopted children ages 14, 13, and 11, know how costly it is and are working to help other families adopt.\nIn November 2014, Bethany and Jared Crain got a call saying that they were matched with an expectant mother.\nWhile they were ecstatic that they could be adding to their family of three, they decided not to tell their 4-year-old that she could be getting a sibling.\n"It\'s just so extremely expensive," said McKay. "A lot of people don\'t have that kind of money."\nThe need on the part of children to have families is great, too. The fund estimates there are 140 million children around the world who are orphaned and 500,000 in the U.S. who are living in foster care and have no permanent family to call their own.\nThe goal of the fund is to inspire adoption by providing grants to qualified parents. Parents who seek grants are required to show that they have financial need and have already completed some of the steps needed to adopt such as working with a licensed agency and having a home study done.\nThe fund puts an emphasis on completing adoptions for children who may find it more difficult to be placed with a permanent family. Those children may be siblings, like Regina and Regino Mapes, children who are aging out, have medical needs or who may go into foster care.\nThe grants that are supplied by the nonprofit range from $3,500 to $7,500.\nJoan Schoon\'s desire to be a foster parent was born out of compassion and sympathy.\nAs a teenager, she remembers listening to her friends who were foster children complain about treatment they endured in some of their previous foster homes.\n"They apply for the exact dollar amount they need and the grants are paid directly to the adoption agency," McKay said.\nThe fund was founded in 1996 by a couple who, like the McKays, felt thankful that they had the financial resources to pay for adopting children, and wanted to help other families who need the support, according to its website.\nShe said the group raises money through a variety of fundraisers such as cocktail parties and golf outings.\n"It\'s such a compelling cause," said McKay, noting that many of their donations are in the $50 to $100 ranges.\n"It\'s a perfect fit," she said.'
            #     },
            #     {
            #         "text": "In previous seasons, K-State developed an effective scouting routine leading up to its NCAA Tournament opener. Video and other information on an opponent began flowing.\nBut the Wildcats will prepare differently this time. On top of adjusting to the approach of a new coaching staff, they will have to spend the next two days planning for multiple opponents before locking in on either La Salle or Boise State on Thursday in Kansas City.\nScouting for two teams instead of one will be a challenge. Coaches will come up with two game plans, and the staff will provide players with twice as much video. To help ease the process, Bruce Weber said four coaches will help gather information instead of the usual three.\nSome will argue preparing for multiple opponents puts K-State at a disadvantage compared to other highly seeded teams. Three other teams in the field of 68 have to deal with the same time crunch and challenge of facing a team that has already won a game in the tournament.\nBoise State averages more than 73 points behind Anthony Drmic (17.3 points) and Derrick Marks (16.3). The Broncos won nine games in the Mountain West Conference — which this season has college basketball’s top conference RPI — and beat Creighton on the road. They are not easy to prepare for.\nNeither is La Salle, which won 11 games in the Atlantic 10 behind dynamic guards Ramon Galloway and Tyreek Duren. The Explorers beat Butler and VCU this season.\nOf course, others will say it is an advantage. Sure, K-State doesn’t know who it will play on Friday, but La Salle and Boise State aren’t even thinking about the Wildcats yet.\nA live game, especially in the NCAA Tournament, can often reveal more about a team than what can be found from replays of regular-season games.\nKansas State’s basketball game against current No. 1 Gonzaga next season at Intrust Bank Arena in Wichita will be Dec. 21, the schools announced Monday. It’s a return trip to Kansas after Gonzaga’s victory in a neutral-court game in Seattle this season.\nK-State season-ticket holders and Ahearn Fund members will have first opportunities to purchase tickets, with a public sale beginning at 10 a.m. Sept. 13 through selectaseat.com or by calling 855-755-7328. Tickets will range between $12 and $200, and K-State students can purchase $10 student tickets in the fall."
            #     },
            #     {
            #         "text": "The Hawaii man who was fired after issuing the false ballistic missile alert in mid-January told reporters Friday that he was very upset over the incident but remained adamant that it appeared, at the time, to be a real-life attack.\nThe former state employee – a man in his 50s who asked to remain anonymous for his safety – said that he was “100 percent sure” that the drill on Jan. 13 was real, even though several other employees heard the word “exercise” repeated throughout the message, according to officials.\nOnce the man realized what had happened, he said he felt like he’d sustained a “body blow.” Since then, he’s reportedly been distressed to the point that he has had trouble eating and sleeping.\nDuring a news conference on Tuesday, investigating officer Brig. Gen. Bruce Oliveira shared details of the state’s probe and said when the mistake was realized, the employee “froze” and “seemed confused.” Another employee had to resume his duties and send a correction message, Oliveira said.\nThe employee also reportedly had at least two previous incidents in which he mistook drills for real-world events, including for tsunami and fire warnings. But Oliveira said the employee was previously counseled and allowed to resume his duties.\nFollowing the event, the employee was fired and Vern Miyagi, who oversaw the Hawaii Emergency Management Agency, took responsibility and resigned.\nToby Clairmont, the agency’s executive officer, also resigned before disciplinary action could be taken, and authorities are in the process of suspending another worker without pay.\nBy 8:20 a.m., Hawaii EMA tweeted there was “NO missile threat” to the state, but failed to send a phone alert for another 38 minutes, causing mass panic among people who weren’t able to check social media.\nFox News’ Katherine Lam and The Associated Press contributed to this report."
            #     },
            #     {
            #         "text": "Wydad Casablanca of Morocco will begin the defence of their African Champions League title against either Mali's Stade Malien or newcomers Williamsville AC of Ivory Coast.\nAs the reigning champions, Wydad are one of five teams to be given a bye into the second round of the tournament, which begins in March.\nBeaten finalists in 2017 Al Ahly of Egypt and DR Congo's Confederation Cup winners TP Mazembe are also straight into the second round.\nThe other teams to skip the first round are the 2016 African champions Mamelodi Sundowns from South Africa and Tunisia's Etoile du Sahel, who lifted the trophy in 2007.\nThe 16 winners of the second round ties will advance to the group stage of the tournament.\nFor the first time ever Zambia had two teams in the draw with Zanaco, who reached the group stage in 2017 drawn to play Gambia Armed Forces, while Zesco United will play Zanzibar's JKU SC.\nA change in the calendar for the Confederation of African Football means that the next Champions Leagues will begin in December 2018 and run through to May 2019.\nAfter the 2019 final the competition will be held from August or September through to May of the next year."
            #     },
            #     {
            #         "text": "The write-up “Doctor had thyself” (Spectrum, October 4) took up the issue of professional ethics among doctors. Once sacrosanct Hippocratic Oath has been obscured by the lure of lucre and commission culture.\nAll human concerns and considerations are at stake so much so that each patient is considered a milch-cow. But all this is antithetical to the concept of a welfare state. Healthcare matters more than anything else in India.\nThe government, intelligentsia, law enforcing agencies and charitable organisations must rise to the occasion and curb illegal and undesirable medical practices. The marketing of drugs should be strictly regulated and supervised. Justice should be prompt and deterrent.\nThe role of a doctor in society ought to be consoling, sustaining and elevating in order to revive the erstwhile cordial and courteous bonds in doctor patient relationship. Still a roaring practice laced with milk of human kindness, credibility and self-esteem will bring fame, prestige and money. Introspect deeply and act resolutely.\nVaranasi or Banaras (Spectrum, October 11) was one of the six flourishing places in the days of the Buddha. British resident, Jonathan Duncan established a Sanskrit college there in 1792. Mrs Annie Besant, an activist of the Theosophical Society started Central Hindu School in 1889, which eventually developed into the Banaras Hindu University in 1915.\nWhen the celebrated Vishvanath Temple in the city was demolished and a mosque was built there under the orders of Aurangzeb, poet Chandar Bhan satirically said: “Ba-been karaamat-e-butkhaana-e-mara ai Shaikh/Agar kharaab shavad khaana-e-khuda gardad” (See the miracle of my temple. Even after its destruction it remains the abode of God).\nPeerless poet, Mirza Ghalib, visited Varanasi on December 1, 1827. He was so much enamoured with the place that he stayed there for about a month. In his poem Chiraag-e-dair (lamp of temple) he admired the city. It comprises 108 couplets, a lucky number for the Hindus. Their rosaries have 108 beads. The poet, who described Banaras as the Ka’aba of Hindustan, says, a wise man told him that doomsday would not come, as God did not want the destruction of this elegant city.\nThe practice of giving English titles to Hindi movies (“Desi movies English titles”, Spectrum, Sept 27) is not new, as many movies have had English titles throughout the history of Indian cinema. First and foremost comes to mind films like Street Singer and President which had K.L. Saigal as hero. Mother India (1957) made by Mehboob is considered a landmark in Indian cinema.\nGuru Dutt and Madhubala came up with an evergreen musical comedy Mr & Mrs. 55. Raj Kapoor produced “Boot Polish” (1956) bringing out the struggles of street children. Another one was Love in Shimla (1959) which introduced Sadhana as a new face. Evergreen hero Dev Anand starred in many movies with English titles like Taxi Driver, House No. 44, Paying Guest, CID, Love Marriage, Gambler, Jewel Thief and above all his magnum opus Guide (1966).\nIn “Road to happiness” (Saturday Extra, Sept. 12) the writer has beautifully enumerated eight points to achieve happiness, which has become a rare commodity in this materialistic and selfish world.\nI fully endorse his points. Nathaniel Cotton’s verse, which the writer quoted to buttress his points, was full of wisdom, prudence, reason, sanity and practical knowledge.\nIt is said that sympathy is a heavenly quality and should be shown to everyone in trouble to attain happiness. Kindness, goodness and loving care of one’s aged and ailing parents, contentment and peace of mind, the belief in “live and let live”, “let bygones be bygones” and practice of ahimsa (non-violence) are the key ingredients of happiness. Noble deeds, good food, good thoughts, good conduct devoid of envy, jealousy, rivalry, grudge, malice, back biting and ill-will lead to happiness. The recital of god’s name acts as an icing on the cake.\nLife is a precious gift of God. It is worth living with all its frustrations, impediments and failures. Those who live it as it comes along can solve problems; overcome hardships to achieve their goals and happiness. One should work and not remain idle to be happy. Bad habits like drinking in excess, smoking, taking opium and other such vices should be shunned as these ruin one’s happiness, home and hearth. Punctuality, the mark of civilisation and culture, must be cultivated to gain happiness.\nThinking about common good rather than about one’s own self, caring more for one’s duties than for rights and providing food, water and shelter to the have-nots can increase one’s happiness manifold. To conclude: Happy is the man, whose wish and care, a few paternal acres bound, content to breathe his native air, in his own ground."
            #     },
            #     {
            #         "text": "A spokesman for the Red Crescent, Mahmoud al-Saadi, said Israeli forces opened fire on two young men who were traveling on a motorcycle near the Jalameh checkpoint in the northern city of Jenin.\nThe two men were transferred to Jenin hospital. One succumbed to his wounds after he was shot in the head with a live round. He was identified by the Palestinian health ministry as 19-year-old Abdullah Tawalba.\nThe other unnamed man is in stable condition.\nThe Israeli army said its troops fired at “two assailants who hurled an explosive device” in the vicinity of the checkpoint, adding no soldiers were hurt."
            #     },
            #     {
            #         "text": "John Beilein and the Michigan Wolverines are one win away from a national title. On Saturday the Wolverines knocked off Loyola-Chicago 69-57, outscoring the Ramblers 47-28 in the second half.\nBeilein, 65, was born and raised in Burt, NY, a small hamlet in Newfane. Beilein began his coaching career at Newfane High School, where he spent three years. He then took over the basketball program at Erie Community College. After a short stint with Nazareth College and nine years at Le Moyne, Beilein returned to Western New York, where he coached Canisius for five seasons. Beilein then spent time at both Richmond and West Virginia before accepting the Michigan job in 2007.\nMichigan hasn't won a national title since 1989. This will be Beilein's second championship appearance with the Wolverines, falling to Louisville in 2012-2013."
            #     },
            #     {
            #         "text": "It’s amazing what a difference time can make in the way we perceive certain technologies. A few years ago, the idea of the cloud was terrifying to many people. What exactly did it mean to have something “in the cloud”? Was your personal information just floating around somewhere, ready for people to steal? Now, a lot of those hesitations have disappeared as people learned more about the cloud and its benefits. Businesses and individuals are more than willing to stick their information “in the cloud,” but it’s really not that simple.\nSure, there are plenty of cloud providers in the market, but choosing the right one for your organization can be a tricky process. Brand name cloud providers are often seen as a reliable option for companies, but the truth is that they may not be the right option for your small-to-medium-sized business (SMB) needs.\nMigrating successfully to the cloud requires a SMB to have the right partner by its side, especially if it lacks cloud expertise or the time to develop an appropriate migration strategy. The right partner will be able to ensure that a SMB’s cloud deployment meets regulatory requirements, offers future scalability and flexibility, and/or provides it with the most cost-effective option possible.\nAgain, finding a cloud provider that can do all of the above should be easy, but that’s not always the case. A lot of cloud providers are focused on helping enterprises rather than SMBs. Luckily, there are some cloud providers that cater specifically to SMB customers.\nIf you’d like to learn more about SMB cloud migrations and how to choose the right cloud provider to meet your company’s needs, be sure to register for the upcoming webinar titled “How to Know if Your Cloud Decision is Right for Your SMB.” The webinar, presented by Ed Dryer, senior technology strategist at Steadfast, will take place on Tuesday, August 22, 2017 at 2:00 p.m. EDT.\nAttendees will gain a better understanding of public, hybrid and private clouds for SMBs; learn the operational and cost benefits of virtualization; hear what is driving SMBs today to make cloud investments; and understand how cloud services complement business continuity and disaster recovery (BCDR) services, managed security, and on-demand infrastructure. If this sounds like something you’d be interested in, you can REGISTER HERE."
            #     },
            #     {
            #         "text": 'President-elect Barack Obama’s transition team held an hour and 15 minute meeting on Tuesday with just over a dozen social justice groups that presented what they see as the concerns of Catholics. In response, some Catholic bishops and commentators have told CNA that they don’t believe these groups’ concerns resonate with those of the Church.\nThe discussion between the Obama transition team and the different representatives touched on international development and trade, health care reform, reducing abortions, immigration, domestic policy and poverty reduction, and the environment.\nThe meeting of the 14 different organizations was organized by Catholics in Alliance for the Common Good and the lobbying group “Network,” which describes itself as "a progressive voice within the Catholic community" that lobbies Congress on justice and peace issues.\nSr. Simone Campbell, director of Network, told the National Catholic Reporter that the meeting was called to "acknowledge the work that some of the Catholic groups had done in the Catholic community during the election and to begin to develop relationships for ‘post-Jan. 20,’ when the new administration takes over after Obama’s inauguration."\nJames Salt, Organizing Director of Catholics United, explained to CNA that Catholics United participated in the meeting by highlighting "key policies that are important to Catholics.\n"Specifically we want the new administration to take seriously its commitment to reduce abortions in America. People of goodwill from both sides of the conversation can agree that 1 million abortions a year are 1 million abortions too many. We wanted to make sure that the Obama administration knew this was one of our highest priorities."\nYet, when Salt was asked if Catholics United planned to hold Obama accountable for his pledge to work to reduce abortions, he was cautious. "We\'re hopeful that the Obama administration is with us on abortion reduction. We were not there to make asks, but rather to build consensus around real solutions."\nSalt also added that no one raised the issue of Obama overturning the Mexico City Policy, which prevents American aid from going to those who counsel women on the availability of abortion.\nAlexia Kelley, Executive Director of Catholics in Alliance for the Common Good, informed CNA that there are "many efforts underway and planned" to show support for the incoming administration as well as to challenge it to keep its abortion reducing commitment.\nAdditionally, Kelley mentioned that the topics of how to help the poor, homeless, children and the sick during these times of economic hardship were also raised.\nBoth Salt and Kelley confirmed to CNA that there was no one officially representing the Catholic Church present at the meeting, although they thought that an Obama team representative had met with key bishops at the USCCB.\nBishop Thomas Wenski, a member of the U.S. Bishops’ Committee on International Justice and Peace, reacted to the meeting by saying, "while the Obama transition team is free to meet with anyone they wish…the fact is that the only ones who speak for the Catholic Church are the bishops.\n"If the transition team wished to telegraph a message that their intention is to marginalize the bishops then there is reason for some serious concern regarding the relationship between the future Obama administration and this nation\'s 60 million Catholics," Wenski said.\nCatholic scholar and author George Weigel expressed his doubts about the meeting’s make up. "If the Obama transition team thinks that meeting with the refugees from the Catholic revolution that never was is a way to open a dialogue with the Catholic Church in the United States, they\'re far less clever than I think they are. This strikes me as simply a pay-off to people who, from the Obama campaign\'s point of view, helped with the ground game in 2008."\nThe proof of the social justice groups’ commitment to promoting Catholic concerns will be in "how these ‘Platform for the Common Good’ folks help the rest of the Catholic Church defeat the Freedom of Choice Act and maintain the Bush administration\'s AIDS and malaria-reduction initiatives in Africa, which has helped millions more poor people than any of these groups has ever managed to do," Weigel explained to CNA.\nBishop of Madison Robert Morlino also added that the transition team must do more to dialogue with the Catholic Church. "Recognizing the stark contrast between the positions on abortion of the President-elect and the teachings of the Catholic Church, it would be a mistake for the President-elect\'s transition team to pretend that this meeting satisfied his promise of dialoguing with the Catholic community," he said.\nThe bishop of Phoenix, Thomas Olmsted, also weighed-in on the meeting by addressing what a Catholic organization should be emphasizing. He told CNA that “Being \'right\' on any number of other issues will never outweigh the taking of human life through abortion. It would be my hope that any group calling themselves \'Catholic\' would make this message abundantly clear, and express grave concern over the possibilities that the new administration may increase funding for abortions with public money or even erode conscience protections for Catholic hospitals and healthcare workers."\nFinally, Brian Burch, who heads a group of four lay Catholic organizations in the political, legal, research and educational fields, also expressed misgivings about the ability of the social justice consortium to rein in Obama’s policies.\n"We are pleased to hear that the Obama transition team is interested in talking with Catholics, but caution that such conversations must be weighed against his reported plans on abortion policy, including his Cabinet selections thus far. Specifically, we remain concerned that the new Administration is composed of leading abortion advocates who are preparing to overturn a large number of existing pro-life laws, while proving hundreds of millions of new taxpayer dollars for abortion.\n"The fact that transition officials are consulting a select group of Catholic organizations who supported Obama\'s candidacy is not surprising. Whether these groups, some of whom claim to adhere to Catholic teaching, are able to hold him accountable on the issue of life, remains doubtful."'
            #     },
            #     {
            #         "text": "2 bedroom home with extra room for office or 3rd bedroom. Kitchen features Oak cabinets. Home features wood flooring in living area and large deck for entertaining. Pets negotiable with deposit."
            #     },
            #     {
            #         "text": "Steven Spielberg has Ready Player One officially out in theaters today and Carl's Jr. tried to get in on the marketing fun by renaming their new charbroiled sliders to \"SpielBurgers.\" The legendary director caught wind of the fast food chain's publicity stunt and quickly shut it down after admitting that the sliders tasted \"pretty good.\" Though Carl's Jr. never officially renamed their product, they're pretty happy that they got some kind of reaction out of Steven Spielberg with a semi-endorsement.\nCarl's Jr. posted a bunch of spoof videos on Twitter advertising the name change of their sliders to the \"SpielBurgers\" to help promote Ready Player One and their new product. The videos each take on some of the director's most famous work, from E.T. to Jurassic Park. The fast food chain even decided to take their guerilla campaign to the next level when they had some of the \"SpielBurgers\" delivered to Amblin Entertainment yesterday, which prompted Steven Spielberg to record a response video and post it to Twitter.\nThe fast food chain tweeted over the weekend about the name change in honor of Ready Player One and noted that Steven Spielberg hadn't signed off, but that they \"assume that he's cool with it.\" In addition to the burgers, an employee even left a letter on an Amblin executive's car. After Carl's Jr. delivered the \"SpielBurgers\" to Amblin, the director announced a cease and desist to the fast food chain after sort of complimenting the taste of their creations. Spielberg had this to say.\n\"It has recently come to my attention that Carl's Jr. wants to rename their charbroiled sliders SpielBurgers and they're pretty good, but I'm passing. Cease and desist. You can't do it - sorry, guys.\"\nThe publicity stunt worked out perfectly for Carl's Jr. They knew that Steven Spielberg would never sign off on the deal, but if they got any response from him, it would be well worth it. Not only did Spielberg respond, he recorded a video that stated that \"SpielBurgers\" were pretty good, resulting in free promotion for the fast food chain and Ready Player One. Obviously, Carl's Jr. is pretty happy with the outcome that they orchestrated. Though the note on the executive's car was a tad bit creepy.\nAs for official partnerships, Ready Player One teamed up with the extremely popular HQ Trivia app for a record breaking prize of $250,000. The trivia game will take place tonight, March 28th at 6:30 PM Pacific and 9:30 PM Eastern, to celebrate the release of the movie. As for Steven Spielberg actually eating Carl's Jr., that has to be a bit of a white lie. One cannot imagine one of the most legendary directors of all time eating that type of fast food, even though our president mows down McDonald's more than once a week. You can check out the brilliant marketing plan to promote fast food and Ready Player One below, courtesy of the Carl's Jr. Twitter account."
            #     },
            #     {
            #         "text": "Capital abundance, low interest rates, and high volatility are creating new challenges and opportunities in equity markets. To succeed in this crowded and complex global landscape, you must take your investment expertise to a new level. Private Equity and Venture Capital, an Executive Education program at Harvard Business School, explores cutting-edge industry models and related issues—from venture capital, growth equity, and buyouts to industry infrastructure, portfolio strategies, and decision-making processes.\nDesigned to improve your effectiveness at all stages of a deal, this program examines innovative approaches to asset management, financial strategy, organizational structure, and portfolio management. You’ll learn how to improve your negotiation approach, identify solid investment opportunities, manage asset inflation and bubble risks, and generate long-term returns to secure a competitive advantage for your company."
            #     },
            #     {
            #         "text": "Chee Wei Wong, professor of electrical and computer engineering in the UCLA Samueli School of Engineering, was named a fellow of the Society of Photo-Optical Instrumentation Engineers.\nThe society was founded in 1955 to advance light-based technologies and annually organizes and sponsors major technical forums, exhibitions and education programs around the world. Fellows are members who have made significant scientific and technical contributions in the fields of optics, photonics and imaging. Wong was recognized for his achievements in ultrafast optics, nonlinear photonics, quantum optics and precision measurements.\nPhysical and wave electronics are Wong’s primary area of focus. Wong is the recipient of the 2018 National Institutes of Health’s Early Scientist Trailblazer Award and the 2016 Google Faculty Research Award."
            #     },
            #     {
            #         "text": "Seren Rayne Frank Sutherland, a six lb., eight oz., girl, was born Saturday, Dec. 3, 2016, at Yavapai Regional Medical Center to Donell Sutherland and Adam Frank of Prescott.\nAlexander Velasco, a six lb., 12 oz., boy, was born Wednesday, Dec. 7, 2016, at Yavapai Regional Medical Center to Erika Avitia Villalobos and Eduardo Velasco of Prescott Valley.\nTeagan Mikelynn Scotia Walls, a seven lb., 11 oz., boy, was born Tuesday, Dec. 6, 2016, at Yavapai Regional Medical Center to Jamie Ashlyn and Michael Scott Walls of Prescott Valley.\nOwen Matthew Wederski, an eight lb., two oz., boy, was born Sunday, Nov. 27, 2016, at Yavapai Regional Medical Center to Kayla Greseth and Joshua Wederski of Prescott Valley.\nRorik Isaiah Wilson, a eight lb., three oz., boy, was born Thursday, Dec. 22, 2016, at Yavapai Regional Medical Center to Marina Wilson and Colton McKeever of Prescott Valley.\nSayge Elijah Anthony Zamora Gheiler, a seven lb.,two oz., boy, was born Friday, Dec. 23, 2016, at Yavapai Regional Medical Center to Nicollette Gheiler of Prescott."
            #     },
            #     {
            #         "text": "Bravo, France! And shame on you, David Cameron!\nFrance has offered asylum to the Iraqi Christians forced to flee from Mosul. The BBC reports this, and so does Al Jazeera.\n“We are in constant contact with local and national authorities to ensure everything is done to protect them,” both ministers said.\nSo, these are not cheap words, or political posturing. Something is actually being done.\nBravo, France! You are a secular republic that sees, in true secular fashion, the human needs of people in distress, and wants to do something about it.\nBravo, France! You have form in this matter already. For France it was that received thousands of refugees from Russia in the aftermath of the revolution there, and also took in thousands of Armenians who survived the Ottoman genocide of 1915. Now, once more, you are helping those who need a safe haven.\nBravo, France! You have expressed outrage at the treatment of the Christians of Mosul, and you have not taken the line that these are merely one oppressed group among many: there has been no ‘universalise to minimise’ strategy here.\nItaly and the Vatican acted over the case of Meriam Ibrahim; France is now prepared to act over the persecuted Christians of Mosul. (Entry to France may well give them entry to the entire European Union.) Over to you, David Cameron and William Hague."
            #     },
            #     {
            #         "text": "Hands-on with a transparent 3D TV Jump to media player Chinese electronics firm HiSense shows off a transparent television at the Consumer Electronics Show in Las Vegas.\nOnline plants a growth area? Jump to media player Theodore Sean goes through Parrot's Flower Power app, a program and tool that lets gardeners put their plants online.\nNY fireman introduces 'life saver' app Jump to media player Charismatic former New York fireman Billy O'Connor tells the BBC why he think his company's mobile protection app is vital to stay safe.\nWill.i.am 'excited by what's not there' Jump to media player Musician-turned-tech entrepreneur will.i.am talks to the BBC about what excites him at the 2013 Consumer Electronics Show in Las Vegas.\nCan 'goo' protect your phone? Jump to media player A firm showcases a \"goo-like\" material for protecting mobile devices at the Consumer Electronics Show in Las Vegas.\nChinese electronics firm HiSense is occupying the space at CES normally reserved for Microsoft.\nThe company is making the most of the Windows-maker's absence by displaying its wares - including this transparent 3D television.\nThe company says it could potentially be used by museums and other attractions to create exciting displays that combine real objects - such as artefacts - with 3D imaging.\nHisense spokesman Payton Tyrell said the screen was still a prototype.\nGo to next video: Online plants a growth area?"
            #     },
            #     {
            #         "text": "The democratization of data is a real phenomenon, but building a sustainable data democracy means truly giving power to the people. The alternative is just a shift of power from traditional data analysts within IT departments to a new generation of data scientists and app developers. And this seems a lot more like a dictatorship than a democracy — a benevolent dictatorship, but a dictatorship nonetheless.\nThese individuals and companies aren’t entirely bad, of course, and they’re actually necessary. Apps that help predict what we want to read, where we’ll want to go next or what songs we’ll like are certainly cool and even beneficial in their ability to automate and optimize certain aspects of our lives and jobs. In the corporate world, there will always be data experts who are smarter and trained in advanced techniques and who should be called upon to answer the toughest questions or tackle the thorniest problems.\nLast week, for example, Salesforce.com introduced a new feature of its Chatter intra-company social network that categorizes a variety of data sources so employees can easily find the people, documents and other information relevant to topics they’re interested in. As with similarly devised services — LinkedIn’s People You May Know, the gravitational search movement, or any type of service using an interest graph — the new feature’s beauty and utility lie in its abstraction of the underlying semantic algorithms and data processing.\nThe problem, however, comes when we’re forced to rely on these people, features and applications to decide how data can affect our lives or jobs, or what questions we can answer using the troves of data now available to us. In a true data democracy, citizens must be empowered to make use of their own data as they see fit and they must only have to rely apps and experts by choice or when the task really requires an expert hand. At any rate, citizens must be informed enough to have a meaningful voice in bigger decisions about data.\nThe good news is that there’s a whole new breed of startups trying to empower the data citizenry, whatever their role. Companies such as 0xdata, Precog and BigML are trying to make data science more accessible to everyday business users. There are next-generation business intelligence startups such as SiSense, Platfora and ClearStory rethinking how business analytics are done in an area of HTML5 and big data. And then there are companies such as Statwing, Infogram and Datahero (which will be in beta mode soon, by the way) trying to bring data analysis to the unwashed non-data-savvy masses.\nCombined with a growing number of publicly available data sets and data marketplaces, and more ways of collecting every possible kind of data — personal fitness, web analytics, energy consumption, you name it — these self-service tools can provide an invaluable service. In January, I highlighted how a number of them can work by using my own dietary and activity data, as well as publicly available gun-ownership data and even web-page text. But as I explained then, they’re still not always easy for laypeople to use, much less perfect.\nStatwing spells out statistics for laypeople.\nCan Tableau be data’s George Washington?\nThis is why I’m so excited about Tableau’s forthcoming IPO. There are few companies that helped spur the democratization of data over the past few years more than Tableau. It has become the face of the next-generation business intelligence software thanks to its ease of use and focus on appealing visualization, and its free public software has found avid users even among relative data novices like myself. Tableau’s success and vision no doubt inspired a number of the companies I’ve already referenced.\nAssuming it begins its publicly traded life flush with capital, Tableau will not just be financially sound — it will also be in a position to help the burgeoning data democracy evolve into something that can last. More money means being able to develop more features that Tableau can use to bolster sales (and further empower business users with data analysis), which should mean the company can afford to also continually improve its free service and perhaps put premium versions in the hands of more types of more non-corporate professionals for free.\nTableau is already easy (I made this) — but not easy enough.\nThe bottom-up approach has already proven very effective in the worlds of cloud computing, software as a service and open-source software, and I have to assume it’s a win-win situation in analytics, too. Today’s free users will be tomorrow’s paying users once they get skilled enough to want to move onto bigger data sets and better features. But the base products have to be easy enough and useful enough to get started with, or companies will only have a lot of registrations and downloads but very few avid users.\nAnd if Tableau steps ups its game around data democratization, I have to assume it will up the ante for the company’s fellow large analytics vendors and even startups. A race to empower the lower classes on the data ladder would certainly be in stark contrast to the historical strategy of building ever-bigger, ever-more-advanced products targeting only the already-powerful data elite. That’s the kind of revolution I think we all can get behind.\nFeature image courtesy of Shutterstock user Tiago Jorge da Silva Estima.\nGreat article Derrick – appreciating your work on the topic here on GigaOm.\nWe’re seeing wider availability of reasonably priced BI and visualization software tools to help us understand that harnessing all this data is possible – and I think even consumers are beginning to understand the value of all the data, and the ability to make meaning from it. One part of the puzzle that’s missing from what I can see is the education – knowledge transfer of how individuals can use the tools, what good data science methods are, and how data citizens can actively contribute to the larger data analysis community. I see movements like the Open Data/Open Gov folks, and events like the NYC Big Apps hackathon as part of the solution – but as individuals, where do we go to take part? What is the role of an informed, curious citizen in this? More venues exist for learning some of the ‘how’ to make sense of big data as an individual taking a course online, but I’m not seeing a vision from anyone talking about how to connect all of the dots. To make sense of data, we need the tools, the practitioners, the analysis of the problems, but we also need a vision of how all of these will work. If anyone has ideas of who’s got that vision, I’d love to hear it.\nI feel one of the biggest impediments to the democratization of data is access. Most people know what they would like to answer, and how the data needs to be shaped to achieve that, but getting the data to do the actual analysis with can be one of the most difficult aspects.\nThis is a bit of a plug, but we’re working on enabling data access that is easily attainable by everyone. Our platform http://www.quandl.com is a “search engine for data” that is able to fetch time series data from a disparate sets of sources, and provide it in a simple searchable form that allows users to extract, validate, format, merge, graph, and share it however they want.\nBy providing the underlying data for analysis tools like Tableau, Statwing, and many others, we feel we can help to create the tool stack that empowers people to create a sustainable DIY data culture.\nIn every company I’ve worked at, I’ve seen this major divide between IT analysts and Business users. Part of it was cultural, but a major reason was as you point out: “a historical strategy of building ever-bigger, ever-more-advanced products targeting only the already-powerful data elite”. The business user typically was left to use Excel to prepare and analyze data.\nIt took 15+ years, but thanks to new players like Tableau, Spotfire and Qlikview which were sold primarily to the business user and focused on ease of use, the data democratization process has resulted in a power shift to the business user. Some IT departments have now come around and are trying to accommodate these “shadow IT” projects by providing IT support and giving Tableau users limited access to enterprise data stores.\nAs for upping the ante for the traditional players, it has happened already. Over the last two years, the larger vendors have responded with products like Visual Insight (MicroStrategy), Visual Intelligence (SAP), PowerPivot (MicroSoft), JMP (SAS) etc. taking aim at this segment of the market. The Big Data market is still new, but the trend to build user-friendly (or at the very least, SQL-aware) tools on top of Hadoop is also hitting its stride.\nOne good thing coming out of this data democratization is the realization that it has to be supported by a Data Governance effort. Otherwise we’ll see the unfortunate return of a major problem with data democracy: data chaos. Previously it would have meant comparing and reconciling two Excel spreadsheets, now we may end up reconciling the findings from two Tableau workbooks.\nThanks for the comment, and for making a really good point about data governance. Obviously, that’s not too big a concern for personal data use, but competing findings from lots of disparate data sets would be problematic."
            #     },
            #     {
            #         "text": "Maddie Hinch and her England team-mates should be proud of their Hockey Women’s World Cup exploits, according to their head coach.\nThe West Chiltington-based goalkeeper’s dreams of adding another medal to her collection were extinguished last Thursday night as they were knocked out at the quarter-final stage.\nThe hosts nation suffered a 2-0 defeat to defending champions Netherlands at the Lee Valley Hockey Centre in London.\nLidewij Welten opened the scoring for the Dutch in the first quarter and Laurien Leurink then doubled the lead early in the second half as the world number one side dominated.\nThe Dutch went on to beat Australia in the semi-finals and then overcame shock finaliasts Ireland with a crushing 6-0 victory in Sunday’s final to retain their crown.\nIt wasn’t the tournament England were expecting after drawing their opening two Pool B matches. They then beat Ireland to finish second in the group, but had to beat South Korea in the crossover game to make the main knockout stages.\nHead coach Danny Kerry, however, defended his team’s performance and pointed to the difficulties they had faced.\nHe said: “I told the players I was really proud. We have had a tough tournament with injury and for all sorts of reasons.\n“One of our players played the entire tournament with a broken big toe but they all grit it out and carried on.\nHinch won Olympic gold on 2016 and Commonwealth Games bronze in Australia earlier this year."
            #     },
            #     {
            #         "text": '"Whoever gets him, they\'ll be getting a good one," David Montgomery said.\nINDIANAPOLIS — Hakeem Butler has been surrounded by some of the best wide receivers on the planet this week at the NFL Scouting Combine.\nIt’s an experience that might humble some. But for Butler, it has only enhanced his confidence.\nAs it stands, 22-year-old Butler is not regarded as the best wide receiver in this year’s NFL Draft. He’s projected by some experts to go as late as the third round. But when wide receivers were measured Thursday, Butler gained some attention: He led all receivers in height (6-foot-5 3/8), arm length (35 1/4 inches) and wingspan (83 7/8 inches).\nOn Thursday, running back David Montgomery, who played with Butler at Iowa State, captured the general vibe surrounding Butler here.\nButler says he’s met with every NFL team on an informal basis. He had “nine or 10” formal meetings set up for Friday night, but didn’t divulge which teams he’d be sitting down with.\nThere is clear interest in Butler, who declared for the draft after his junior season in which he had 60 receptions for 1,318 yards and nine touchdowns.\nBut in his mind, the hype machine hasn’t been turned up high enough — yet.\nButler, of course, is talking about Saturday’s wide receiver workouts. If he crushes the drills, he could vault up the draft board — perhaps into the first round. And he feels well-prepared because he’s spent the past few months working out with some legendary NFL receivers in Calvin Johnson and Anquan Boldin.\nButler met Boldin at the South Florida gym where he trains. He was connected to Johnson through his agent.\nJohnson and Boldin are known for being precise, tactical receivers. But Butler says the most valuable lesson of working with them has been learning about how they think.\nButler admitted the first time he worked with Johnson, who went to six Pro Bowls with the Detroit Lions from 2010-15, that he was a little starstruck.\nButler is hoping that one day he can leave a legacy like Boldin or Johnson. But for now, his goal is to prove what he already believes is true: that he is the top receiver in this draft.'
            #     },
            #     {
            #         "text": "Brittney Griner had a pretty average game last night against Florida—by her standards, anyway—when she finished with 25 points, nine rebounds, six blocks, and four assists during Baylor&apos;s 76-57 win. But she also did something that was definitely not average when she threw down a dunk at the start of the second half to become just the second woman in the history of women&apos;s basketball to dunk during the NCAA Tournament. As if we needed any further proof, Brittney Griner is officially a beast."
            #     },
            #     {
            #         "text": "Liberty Hill is conveniently located in the vibrant city of McKinney, named one of the Best Places to Live in America, near 121 and 75.. You won’t have to venture far with multiple corporate headquarters, shopping, and dining just minutes away. Normandy Homes features both single story and two story homes in Liberty Hill with a mix of Traditional, Tudor, and French Country inspired exteriors. Children will attend the highly acclaimed Frisco ISD."
            #     },
            #     {
            #         "text": "McDonald’s jumped 5.3 percent after the world’s biggest fast-food chain by revenue topped analysts’ forecasts for profit and sales.\nShares of oil refiner Andeavor surged 14.4 percent, the biggest percentage gainer on the S&P 500, after rival Marathon Petroleum agreed to buy the company for more than $23 billion. Marathon’s shares slid 4.2 percent.\n“The big news was really the deals, that continue the trend of strong M&A environment,” said Aaron Clark, portfolio manager at GW&K Investment Management in Boston, Massachusetts.\nWalmart rose about 2 percent after Sainsbury’s agreed to buy the UK arm of Walmart, Asda, for about $10 billion, while Marriott Vacations Worldwide said it would buy timeshare operator ILG Inc for $4.7 billion, sending the target company’s shares up 4.5 percent.\nAnother big deal announcement was that of T-Mobile’s $26 billion takeover of fellow wireless carrier Sprint . Sprint shares fell 13.5 percent as analysts said it could face antitrust hurdles and the offer was seen as less favorable than an earlier one.\n“Muted reactions to what was very strong earnings, capex spending picking up and strong M&A applies to what I’d say is part of classic late-cycle behavior,” said Clark.\nAt 11:36 a.m. ET, the Dow Jones Industrial Average was up 125.42 points, or 0.52 percent, at 24,436.61, the S&P 500 was up 4.38 points, or 0.16 percent, at 2,674.29 and the Nasdaq Composite was up 7.74 points, or 0.11 percent, at 7,127.54.\nU.S. bond yields edged lower after data showed March personal income rose lesser-than-expected, and personal spending in February was revised down to 0.3 percent, from the previously reported 0.4 percent.\nOf the 274 S&P 500 firms that have reported first-quarter earnings so far, 79.2 percent topped profit expectations, according to Thomson Reuters data. That has lifted the estimate for earnings growth to 24.6 percent from about 18 percent at the start of the season.\nHealthcare stocks were a drag, led by Celgene’s 6 percent fall after Morgan Stanley noted it expects delay of up to three years for the company’s key multiple sclerosis drug.\nThe S&P healthcare index was was down about 0.6 percent.\nArconic plummeted 15.2 percent after the aluminum products maker slashed its 2018 forecasts for profit and free cash flow as higher prices of the metal squeezed profit margins.\nAllergan Plc reversed course to fall 4.1 percent after its chief executive officer said he was opposed to fundamental changes to the drug company’s business strategy.\nAdvancing issues outnumbered decliners for a 1.16-to-1 ratio on the NYSE and for a 1.04-to-1 ratio on the Nasdaq."
            #     },
            #     {
            #         "text": 'One-time Yankees killer Dallas Keuchel is entertaining the idea of donning pinstripes.\nSo much so the free agent pitcher is willing to shave his signature beard in order to abide with team protocol.\n"I think everybody is in play right now," Keuchel said in a recent interview with Fox Business. "The lure of the city would be really cool. I like pitching in Yankee Stadium.\n"For the right opportunity, I would happily shave this beard off," Keuchel said, channeling his inner-Johhny Damon who did so in 2005. "It\'s all about winning. I\'ve made that very clear from Day 1 of my career starting to this position right now."\nBut don’t stock up on shaving cream just yet, Yankees fans.\nWhile Keuchel, 31, has a history of tormenting the Yankees stretching back to the 2015 wild card game, he struggled a ton this past season.\nThe Yanks, who have made pitching a priority this winter, had their way with the southpaw, tallying seven runs in two wins against Keuchel.\nIt was an overall down year for the 2015 Cy Young winner, who posted career highs in hits allowed, walks and WHIP.\nThe Daily News’ Wally Matthews suggested Brian Cashman stay away in his Yankees free agency primer.'
            #     },
            #     {
            #         "text": "Inter Milan's lead at the top of Serie A was cut when they lost 2-1 at home to Lazio on Sunday while AS Roma beat Genoa 2-0 to record their first win in eight games and ease the pressure on under-fire coach Rudi Garcia.\nThe title race remains congested as second-placed Fiorentina defeated Chievo 2-0 and Napoli, in third, beat Atalanta 3-1. Juventus' winning streak continued when they came from behind to beat Carpi 3-2.\nNapoli and Fiorentina have 35 points, one behind Inter who were sunk by two-goal Antonio Candreva.\nRoberto Mancini's side fell behind when Candreva struck a fifth-minute thunderbolt but they levelled as captain Mauro Icardi slotted the ball beneath Etrit Berisha in the 61st.\nBrazilian midfielder Felipe Melo then gave away a penalty in the 87th minute after fouling Sergej Milinkovic-Savic and Candreva beat Samir Handanovic on the rebound after his initial effort was blocked.\nMelo lost his composure and was dismissed in stoppage time after aiming a kung-fu kick at Lucas Biglia and Milinkovic-Savic received a second yellow card moments later.\n\"Unfortunately Melo did two stupid things,\" coach Mancini told Sky Sport Italia. \"We did the damage ourselves, we threw it away.\"\nMario Mandzukic scored twice as Juve extended their Serie A winning streak to seven matches to head into the winter break in fourth on 33 points, a point clear of Roma in fifth.\nFormer Juve striker Marco Borriello gave second from bottom Carpi the lead but Mandzukic equalised in spectacular fashion when he swivelled to fire a volley past Vid Belec.\nMandzukic netted again with a header in the 41st minute. Paul Pogba added a third before Leonardo Bonucci's late own goal set up a tense finale.\nRoma's Alessandro Florenzi and Sadiq Umar scored, while Edin Dzeko was sent off for swearing at the referee, as Garcia's side ended a run of seven games without a win in all competitions by beating Genoa.\nGarcia was close to the exit door, according to media reports, following their midweek Italian Cup elimination by Serie B Spezia but may have bought himself some time.\n\"We can see the light at the end of the tunnel,\" said Frenchman Garcia. \"We remain close to second place and it's all open for 2016.\"\nNapoli saw off Atalanta with two second-half goals from Gonzalo Higuain although both sides had a player sent off during an ill-tempered match.\nFiorentina downed Chievo with goals from Nikola Kalinic and Josip Ilicic while AC Milan's fine form under Sinisa Mihajlovic continued as they came from behind to win 4-2 at struggling Frosinone to extend their unbeaten run to eight in all competitions.\nVerona and Sassuolo drew 1-1, Sampdoria beat Palermo 2-0 and Udinese won 1-0 at Torino."
            #     },
            #     {
            #         "text": "A three member bench of the Supreme Court has adjourned the hearing of Orange Line Metro Train case till Friday.\nDuring the course of proceedings today, the court summoned the Chairman Planning Commission, the Secretary Finance and the NAB prosecution team on the next hearing.\nIn his remarks, Justice Azmat Saeed said work on the project should not stop because of funds."
            #     },
            #     {
            #         "text": "As usual, Professor Jack Ponton (Letters, 8 July) cuts through all the smoke and mirrors erected by the renewables industries to give the facts.\nHe points out the absurdity of the regular and monotonous claims from the wind industry that at certain times they providing enough electricity to power umpteen million homes.\nUntil there is a research breakthrough in battery storage, wind-generated electricity will continue to be an expensive failed science which only benefits the developers and land owners.\nI am reminded of a former MSP who berated me on my anti-wind stance, stating that the wind produced at night was more than required during the day and was not wasted but stored in batteries.\nI asked her where this scientific breakthrough was located. Silence.\nI suspect that many of our present politicians are equally ignorant and are unaware of the risks that renewables pose to our energy security, especially with their anti-fossil fuels fixation.\nDo they appreciate that Longannet is due to close soon, followed by nuclear?"
            #     },
            #     {
            #         "text": 'Hull - Where the US goes the UK often follows. This is especially true of negatives. If the US has extreme weather we usually experience similar a few days later. If violent crime increases across the Pond likewise in the UK. Then there is shopping madness.\nThe UK is taking up traditional US activities such as Christmas shopping bargain days that leave customers with empty wallets and a range of unwanted goods.\nBlack Friday has just passed and today is Cyber Monday. Each have become big shopping days on both sides of the Pond. Last Friday many people hit the shops attracted by huge price cuts and today the same is happening online. Is all as it seems though?\nAs usual the answer is probably not.\nIt is so easy to get sucked into spending unnecessarily. You buy in haste and repent at leisure. This year remember it is still in reality a consumer market. Businesses want your trade. That does not stop some of them throwing the odd rogue item into the pot.\nThis year more than ever retailers will be trying hard to get you to part with your hard-earned money. There will be pre-Christmas sales and bargains long after today. With Christmas a few weeks away perhaps it would be wiser to hang fire. Shop around, be it online or on the High Street, take your time and purchase well.\nBargains are sometimes far from that. They could be second-rate, last year\'s model or simply stock that is hard to shift. Once it is dressed up with a fancy cut-price sign you may find it hard to resist.\nConsider all the implications of your purchase before you buy.\n1- If you buy online there may be delivery charges but you will be saving a potentially expensive trip to town which could be full of hassle.\n2 - Emails dropping into your inbox, that offer great bargains, can be tempting. However, if you purchase an item you had no intention of buying is it really a bargain? Probably not, especially if money is tight.\nHaving some idea of what you want to buy helps. If you shop "blind" you will be more likely to spend more than you should. Online retail oulets, just like the shops, are full of goods you never knew you wanted or needed and in many cases cannot afford.\nCyber Monday also increases the temptation to buy now and pay later. Not a good idea. Pay for purchases online with a switch or debit card so you are not paying for Christmas well into the New Year or even beyond.\nOne factor specific to the UK is that this week Chancellor George Osborne will announce to the country his Autumn statement, or Budget. It is doubtful that it will have a real positive effect on retail trade but it could. It could also cost you dearly in many ways. Brits take care and look after your money this cyber Monday.\nPerhaps the rest of you would be wise to follow suit too!'
            #     },
            #     {
            #         "text": "A plan to build fifty giant wind turbines in the province of Drenthe may interfere with the operation of a radio telescope and do ‘disastrous’ damage to scientific research, according to the Dutch Institute for Radio Astronomy (Astron).\nAstronomers claim that the placement of the 200 metre high windmills will interfere with the low-frequency array (Lofar) which uses thousands of low-frequency antennae to survey the universe. Because of their height, the windmills reflect other radio and television signals towards the Lofar station.\nMinister for economic affairs Henk Kamp has a year to decide whether to continue with the project. The plan has already been criticized by local residents and municipalities.\nThe Dutch telecom agency is currently looking into whether windmills interfere with radio telescope operation in Drenthe.\nMinister Kamp gave the project the go-ahead based on a study by a British company, which found that the operation of the telescope would not suffer if windmills were placed nearby. However, Astron argues that the British researchers didn’t have sufficient knowledge to make that claim.\nAstronomers at the Lofar site would prefer if the windmills were at least 15km away from the telescope. The current plan would see seven of the fifty turbines placed within a 15km radius of the Lofar site.\n‘It might not sound like too much of a difference, but going from a few kilometres to fifteen really makes a big difference when it comes to the strength of the reflected signals,’ said Garrett."
            #     },
            #     {
            #         "text": "Still image of Lisa Sharon Harper from YouTube.\nPastors and lay leaders who represent minority and multiethnic communities and are appalled by the prospect of a Donald Trump presidency have a blunt message for the white evangelical majority that helped elect him: we’re disappointed in you, but not surprised.\nFor these evangelicals of color, Trump’s use of racially-charged language, his anti-immigrant rhetoric, negative remarks targeting Mexicans and Muslims, as well as the emergence of the “Access Hollywood” tape and his other divisive comments about women, were simply disqualifying.\nWhile some prominent white evangelical leaders made their opposition to then-candidate Trump widely known (many signing a letter protesting his candidacy), the majority of white self-identified evangelicals (estimated to run as high as 81 percent), lined up behind him.\n“Many of [Trump’s] critics fell silent or fell into line, while the group known as the ‘religious right’ continued to support him’ says Kathy Khang, a Christian writer and speaker based in the Chicago area.\nFor the past eight years, people of color, the LGBT community, and women have been given license to flourish, says Lisa Sharon Harper, author of The Very Good Gospel: How Everything Wrong Can Be Made Right and chief church engagement officer at Sojourners. “The white church demonstrated on November 8th that it is more white than Christian, and has a [greater] commitment to white supremacy than it does to Christ,” says Harper.\nThe fact that so many evangelicals didn’t see Trump’s controversial rhetoric as derogatory underlined the presence of a persistent and troubling racial divide in American Christianity that these leaders say is deeply rooted in American history.\nSome are questioning the value of continued association with the white evangelical majority.\nDespite their dismay over the prospect of a Trump presidency, those I spoke to appear to be more motivated and energized than daunted by the challenges that lie ahead.\n“This has been a wakeup call to the progressive, moderate community that we have to stand up for what we believe in and communicate it in the public square,” DuBois concludes.\nAnd Lisa Sharon Harper tells me that “a new Civil Rights movement is happening, and its locus is in people of color.” She sees evidence of it already in the “movement for black lives,” the witness of the so-called “Dreamers” (undocumented immigrants who arrived here as children), and the rising call for solidarity with the poor that mirrors the words of Jesus in Matthew 25. “Every word of Scripture was written by oppressed people,” she says.\nElizabeth Eisenstadt Evans is a Pennsylvania-based freelance writer, and a religion columnist for LNP Media, Inc in Lancaster, PA. Her work has appeared in the Philadelphia Inquirer, the National Catholic Reporter, the Global Sisters Report, Religion News Service and other media outlets."
            #     },
            #     {
            #         "text": 'Tulsa race riot survivor: "Learn how to forgive, but never forget"\nSAN FRANCISCO, California (CNN) -- A 100-inch, high definition screen projects an intense college basketball game. Massage therapists rub the nervous tensions of men and women away. Scissors skillfully cut men\'s hair. Two chandeliers adorn the main room, complimented by brick walls and a glass bar that doubles as a retail counter.\nSean Heywood, right, and Kumi Walker own MR., a barbershop and wine bar in San Francisco, California.\nThis is not your typical barbershop.\nAnd that has always been the vision of owners Kumi Walker and Sean Heywood.\n"We are literally trying to create a new version of the country club golf experience. But we\'re replacing golf with haircuts, and we\'re putting it in urban environments," says Heywood.\nMR. (for mister) is their first business venture. It\'s an upscale barbershop, wine bar and lounge in the financial district of San Francisco where memberships cost $65 to $250 a month. In addition to the basic services, those who become members benefit from exclusive services like golf putting clinics and human resource workshops.\n"We\'re trying to thrive, not just survive," Heywood said.\nMR. takes the cultural aspect of the black barbershop experience and modernizes it, making it a place where businesspeople of all races can network.\nThe two entrepreneurs are bridging a cultural divide, and also giving back to the community. They offer free lifetime memberships to MR. to those in good standing with a re-entry program called Back on Track.\nAmong other things, Back on Track offers first-time, low-level criminal offenders GED preparation, tutoring, money-management instruction and job training and placement. And that\'s where MR. steps in. The membership provides them with one free haircut, trim, and shoeshine monthly.\n"We\'ll take care of their grooming so that they don\'t have to. And they\'re ready for all the different jobs that they pursue going forward," Walker says.\nGiving back is paramount for the entrepreneurs. And the story of one black business district in Tulsa, Oklahoma, inspires them. It was known as Black Wall Street, and it was destroyed in a race riot and fire 88 years ago.\n"All of the businesses that we wanted to create, we wanted to encompass the culture of, if that community still existed today, what it would look like," says Walker, who says he read about the riot six years ago.\nImagine a credit union, a barbershop, a library, and men in freshly pressed suits with top hats sauntering on sidewalks. The melodic sounds of jazz flow into the streets from several nightclubs. A thriving community of black-owned businesses serve their clientele across a 42-block area.\nThat was the community that existed in the segregated neighborhood of Greenwood from 1830 to 1921.\nBut on the evening of May 31, 1921, white mobs entered Greenwood with torches and guns. Black residents gathered to protect a young man accused of assaulting a white woman. When the smoke from fires cleared on June 1, more than 1,000 homes, businesses and other institutions were burned or destroyed, according to the report of the Oklahoma Commission to Study the Tulsa Race Riot of 1921.\nSound off: What challenges do black entrepreneurs face?\niReport.com: How would you make black America better?\n"People came and said, \'Run, they are shooting people,\' " says Wess Young, who was 4 then. "We evacuated. They were destroying everything."\nThe death toll has been debated for years, because many victims were dumped in mass graves. An American Red Cross estimate puts the total at 300, much higher than the 36 reported by local officials.\nThe riot devastated the social underpinnings of the Greenwood community and leveled a black economic force. Greenwood was rebuilt, but it never recovered.\n"It was a really tragic end to thriving businesses. I think we\'d be a lot further ahead had that area been able to continue to thrive," says Walker.\nWess Young and his family rebuilt their lives in Tulsa. When asked what he thought America would be like if the Tulsa race riot had not happened, he answers without hesitation, "We would have had a black president before now. ... He has done a good job, but we [blacks are] still in a box."\nHis advice to people is to let go of the past once they\'ve learned from it.\n"Hate will destroy your whole universe -- got to learn how to forgive, but never forget," Young says.\nWalker says these stories of black struggle and survival motivate him.\n"I stand on the shoulders of my ancestors," he says. "I just want to be as successful as possible so I can turn around and be mentors and sponsors to other people who come after me."'
            #     },
            #     {
            #         "text": "World Series of Fighting on Tuesday announced that Alexandre has inked “an exclusive, multi-year agreement” to compete for the Las Vegas-based organization. Specifics of the deal were not disclosed.\nA decorated muay Thai practitioner, Alexandre began his mixed martial arts career in 2011 and ran up a 5-1 record inside the Bellator cage in the span of 13 months, including a rematch win against the only man to beat him, Josh Quayhagen. The 33-year-old was absent from MMA for more than a year before returning to knock out Rey Trujillo in his most recent bout under the banner of Texas’ Legacy Fighting Championship.\nAlexandre has focused mainly on kickboxing in the past two years, most recently defeating John Wayne Parr for the Lion Fight super middleweight title in October.\nThe date and opponent for Alexandre’s promotional debut “will be announced soon,” according to a release. WSOF has two events on its slate for the end of the year: WSOF 25 on Nov. 20 in Phoenix and WSOF 26 on Dec. 18 in Las Vegas."
            #     },
            #     {
            #         "text": "OKLAHOMA CITY- A metro family is looking for answers after their 21-year-old daughter was found dead.\nPolice are saying the death of Sandra Stevens appears to be a suicide, but those closest to her think someone else might be responsible.\nSandra’s mother says there’s no way her daughter took her own life. She says Sandra had just finished hair school and was working at a local restaurant.\n“She had the most beautiful smile and a twinkle in her eye,” Sylvia Stevens said.\nShe says her daughter was always so full of life.\n“You can see most of her pictures, she was always happy, she had plans for her future,” Stevens said.\nThat future was cut short after what happened inside a northwest Oklahoma City home.\nAbout a month ago, Sandy moved in with her boyfriend of two months.\nOn Dec. 6, police rushed to Sandy's new place to investigate a shooting.\nWhen they arrived, officers found her dead inside the home.\nHer boyfriend told police she killed herself with a shotgun, but her family says she never would've done that.\nSandy went to her parents’ home the night before her death. Her mother says Sandra and her boyfriend were fighting.\nHer parents say she wanted to move back home.\n“She was upset, and my husband told her she needed to finish the relationship,” Stevens said.\nThat was the last time Sylvia saw her daughter alive.\nAllegedly, there were a couple of people other than her boyfriend at the scene when Sandy died.\nWhen NewsChannel 4 went to speak with those alleged witnesses, they were not happy to learn about the story.\n“I want the truth. I want the truth,” Stevens said.\nSylvia says she’ll never give up on getting her questions answered.\nA Facebook page dedicated to her daughter, called “Justice for Sandra Stevens,” has more than 3,000 likes.\n“She loved life, and she knew she was loved. I have faith with all my heart that justice will be made. Justice is going to be for Sandy, justice for Sandy,” Stevens said.\nPolice say this is still an active investigation, adding that they have interviewed Sandy's boyfriend.\nThe family is holding out hope that they’ll be able to piece together what really happened.\nThey requested an autopsy, but the medical examiner’s office says it has not finished the report yet."
            #     },
            #     {
            #         "text": "How Far is Skyhome Mahalakshmi Nagar?\nMahalakshmi Nagar is a residential Plot development by Skyhome Enterprises. It is an Ready to occupy project of Skyhome Enterprises. It has a thoughtful design and is being developed with all the modern day amenities as well as basic facilities."
            #     },
            #     {
            #         "text": "We’ve known about Disjointed, Netflix’s upcoming stoner comedy starring Katy Bates and co-created by Chuck Lorre (Mom, The Big Bang Theory, Two and a Half Men), for nearly a year now, but it wasn’t until Thursday that we were given a first look. In the 19-second teaser cheekily released on 4/20, we see Kathy Bates emerge from a smoke-filled weed dispensary while puffing on a joint. She says nothing, because what else is there to say? This is a show about Kathy Bates working in a dispensary, and that’s more than enough to get people interested.\n[Disjointed] follows Ruth (Bates), a lifelong advocate for legalization, as she finally lives her dream as the owner of an Los Angeles cannabis dispensary. Joining her at Ruth’s Alternative Caring are three charismatic “budtenders” (Dougie Baldwin, Elizabeth Ho and Elizabeth Alderfer), her entrepreneurial twentysomething son (Aaron Moten) and a very troubled security guard who served in Afghanistan (Tone Bell).\nAll 20 episodes of Disjointed will be released on Netflix August 25."
            #     },
            #     {
            #         "text": "MADRID: As Cristiano Ronaldo enjoyed his time off at the French Riviera, his Portugal team-mates were busy handing Italy another disappointing result in the UEFA Nations League.\nPortugal did not need Ronaldo, who skipped international duty to recharge after a busy summer, to beat Italy 1-0 in Europe’s newest soccer competition.\nElsewhere, Turkey mounted a remarkable comeback from two goals down against Sweden, while Kosovo made history with their first competitive win since being admitted to UEFA in 2016.\nThe European champions easily outplayed Italy, which had opened its Nations League campaign with a disappointing 1-1 home draw against Poland in its first competitive match under coach Roberto Mancini.\nAndre Silva scored on a counterattack early in the second half to give Portugal the Group 3 win in the top-tier League A. The hosts had several chances to extend their lead at the Stadium of Light in Lisbon, while Italy created little and relied mostly on set pieces.\nThe Nations League gives UEFA’s 55 member countries competitive games and eliminates friendlies. The winners of the League A groups featuring the highest-ranked countries go into a final-four competition in June.\nSantos did not call up Ronaldo after talking to the player and saying he needed more rest following the World Cup and his transfer from Real Madrid to Juventus. Ronaldo also didn’t play in the team’s 1-1 draw in a friendly against Croatia last week.\nSocial media photos showed Ronaldo and his family enjoying time off in Monaco in recent days.\nIt was Italy’s second competitive match since last year’s World Cup playoff loss to Sweden.\nTwo goals five minutes apart early in the second half gave Kosovo a historic 2-0 win over the Faroe Islands in Group 3 of the fourth-tier League D.\nKosovo, the Balkan republic which was accepted by UEFA and FIFA two years ago, had opened with a 0-0 draw at Azerbaijan.\nTurkey bounced back from an opening loss to Russia by coming from two goals down to defeat Sweden 3-2 thanks to two late goals by Emre Akbaba.\nAkbaba scored in the 88th minute and two minutes into stoppage time to give Turkey the League B victory.\nSweden, making its Nations League debut following its surprising quarterfinal appearance at the World Cup, had taken a 2-0 lead by the 49th minute at Friends Arena in Stockholm.\nIt was Turkey’s first away win in seven matches.\nIn the third-tier League C, Scotland ended its run of three straight losses with a 2-0 win over Albania in Group 1.\nIn Group 4, Serbia and Romania drew 2-2, while Montenegro defeated Lithuania 2-0.\nIn League D, Andorra and Kazakhstan drew 1-1 in Group 1, while Malta drew with Azerbaijan 1-1 in Group 3.\nThe Nations League guarantees at least one of Europe’s low-ranked League D teams will qualify for Euro 2020 through a playoffs."
            #     },
            #     {
            #         "text": "While granting bail, the court said it had dismissed the first plea as the matter was “at the stage of investigation” then but the circumstances had since changed.\nA Delhi court on Tuesday granted bail to Gautam Khaitan, who is being probed by the Enforcement Directorate in a black money and money laundering case.\nSpecial judge Arvind Kumar granted him bail on a personal bond of ~25 lakh. The criminal case under the Prevention of Money Laundering Act (PMLA) was filed by the ED on the basis of a case lodged by the Income Tax Department under provisions of the Black Money (Undisclosed Foreign Income and Assets) and Imposition of Tax Act, 2015.\nED contested Khaitan’s bail plea, arguing that he may hamper the investigation. Khaitan’s counsel Pramod Kumar Dubey and senior advocate Sidharth Luthra said that the investigation in the case was over and ED had already filed a charge sheet on March 25. This is the second bail plea moved by Khaitan, which was dismissed by the court.\nChristian Michel, the alleged middleman in the AgustaWestland chopper scam, moved a Delhi court Tuesday seeking 7-day interim bail to celebrate Easter with his family. Special Judge Arvind Kumar directed the CBI and the ED to respond to his application by April 18, when the court will hear the mater."
            #     },
            #     {
            #         "text": "Residents and businessmen in the Washoe Valley, Pleasant Valley area generally see the Interstate 580 freeway extension as a mixed blessing.\nOnce open, Nevada Department of Transportation officials say as much as 70 percent of the traffic on Highway 395 will move to the freeway, greatly reducing the traffic on the old road.\nChris Jacobsen, who lives in what he described as a luxury home in Washoe Valley, agreed it will be a blessing for the residential areas along the current Highway 395 route. But Jacobsen, a consultant who advises businesses – primarily convenience stores – on where best to locate, said it will overall hurt the businesses in Washoe City. He said that applies especially to convenience stores, the gas station and businesses like the Chocolate Factory and Nevada Lynn Emporium which rely on impulse buyers seeing them and deciding to stop.\nHe said Paul Marazzo, owner of Washoe Flats restaurant – formerly the Cattleman’s – may benefit because his is a destination rather than an impulse stop.\nMarazzo is counting on that. He said when the trucks and other through traffic move to the new freeway, it will also make it much easier and safer for drivers seeking a nice dinner at the restaurant he and his brother, Lynn, operate. And, as the valley develops, he said he’ll get more and more local business.\nAt the same time, he said the freeway will make it easier for people to come to his restaurant because they’ll be able to take the freeway to Parker Ranch Road just south of the restaurant.\nAnd in the meantime, he said the freeway construction crews are excellent customers.\nShe said traffic is the issue and she has been involved in efforts to get people to slow down through the valley.\nTyson Petty, manager of Old Washoe Station, the gas station and mini-mart to the north, made similar comments.\nCouch and Petty both said their businesses may be hurt somewhat but neither thought the loss of traffic would put them out of business."
            #     },
            #     {
            #         "text": "Filed to: Whatcha Gonna Do?Filed to: Whatcha Gonna Do?\nBeing a successful professional wrestler takes more than having the right moves in the ring, and no one knows that better than the legendary Hulk Hogan. That's why the Kinect-powered Hulk Hogan's Main Event focuses as much on winning the crowd as it does winning the match.\nTerry Gene Bollea didn't become one of the world's most iconic wrestlers by knowing how to do an Atomic Drop. He did it by becoming Hulk Hogan, a bombastic, charismatic, and generally larger-than-life personality capable of bending a crowd to his will whether a shining red and gold hero or a black-stubbled villain. The man knows how to put on a show. Hopefully he knows how to put on a game as well.\nIn Hulk Hogan's Main Event, developed by Panic Button for Majesco, the Impact Wrestling superstar takes players' custom characters under his meaty wing, guiding them on the path to stardom. He'll coach them on his signature poses as well as more than 30 wrestling combos using MIcrosoft's Kinect sensor to measure their movements. The more dynamic their motions, the more effective their performance. Once they feel the true power of Hulkamania coursing through their veins players are sure to dominate the game's nine increasingly lavish venues. It even supports two-player tag team matches, complete with virtual metal chairs and ladders, which certainly won't lead to anyone getting seriously hurt.\n\"Listen up, people! You will feel the power of Hulkamania when you step into this game,\" said Hulk Hogan. \"Whether you are taking the damage or selling the pain, this game will let you unleash your inner wrestler as you hype up the crowd while putting the hurt on anyone that stands in your way!\"\nHe's right! Too long have our inner wrestlers been leashed! Too long have our wrestling games delivered the sport without the spectacle! It's time to put on the hurt!\nSee? The guy is really good at that.\nHulk Hogan's Main Event is due out this fall from Majesco. Keep an eye out for more Hulkamania during E3 early next month."
            #     },
            #     {
            #         "text": "Perhaps The View is looking for yet another new co-host…?\nThe syndicated daytime talker The Meredith Vieira Show will come to an end early next year after two low-rated seasons, according to a report by Page Six.\nDespite efforts to bolster the program’s viewership by tinkering with its format, reps for celebrity guests have been informed that the show will be cancelled in March, a source tells the site.\nSeason 2 of the former View co-host’s solo effort opened this past September to a 0.8 household rating, down 50 percent from its very first installment."
            #     },
            #     {
            #         "text": "creator Jeff Miller, Rodman is unhappy with the negative way the video game portrays him and has asked Miller to remove his character from the game.\nThe motion comes after North Korea has been accused of hacking into Sony Entertainment's computers and leaking hundreds of confidential — and extremely embarrassing — emails. The cyber-terrorist attack was the first of its kind and resulted in Sony canceling the release of The Interview, which portrays James Franco and Seth Rogan killing Kim Jong-un."
            #     },
            #     {
            #         "text": "Ljubljana, 24 March - Temperatures will stay below the freezing point even during the day on Monday, except in the coastal region. The feeling of cold will be intensified by the chill of winds and it will snow in most parts of the country.\nThe news item consists of 411 characters (without spaces) or 98 words words."
            #     },
            #     {
            #         "text": "A new development designed by the architects Ron and Jim Vari, 33rd Street Square acknowledges both past and present in rapidly changing Bridgeport. The cutouts and big modular panels give the place a contemporary feel, while the red-brick construction recalls the neighborhood’s traditional housing.\nThe development (at 33rd Street and South Parnell Avenue) also positions every unit to take advantage of the two best local views: the fireworks over U.S. Cellular Field and the city skyline. Each of the 11 units has one upper-floor terrace looking south and another looking north. “You’re never going to want to go inside all summer,” says Jennifer Liu, an Atland Realty agent selling units in the building.\nThe spaces inside are roomy and bathed in light from a two-story gridwork of windows; in most of them, an overlook from one floor to another also helps light reach the lower level. At presstime, three units remained at 33rd Street Square, all ready for occupancy. The lowest-cost unit is a 2,800-square-foot three-bedroom unit priced at $559,000; at the upper end, a 3,400-square-foot corner unit with dramatic cutouts and light wells is priced at $675,000."
            #     },
            #     {
            #         "text": 'Yet again the transfer window seems to be passing by with barely a mention of Stoke City. This summer all the narcoleptic gamblers, crocks, and \'ard workers seem to be tied down to lengthy contracts elsewhere.\nMuch to my son&apos;s continued disgust, Stoke City are the least interesting Premiership club to support during the transfer window.\nAlmost the entire school summer holidays can be wasted parked in front of Sky Sports News watching absolutely nothing happening.\nOr as Sir Anthony Pulis would put it, &apos;waiting for something to drop&apos;.\nFrankly it&apos;s enough to turn a 15 year old into a Sunderland fan, at least for the duration of the window.\nHow exciting that must be: watching players come, go, nearly come, nearly go, pass within 150 miles of the Stadium of Light on their way to visit the in-laws only to find themselves snapped up on double the salary just because they stopped off for a cup of tea somewhere outside Carlisle.\nContrast that with Stoke City; for a start you can discount any player we&apos;re linked with as they will also be linked to several other clubs all of whom will be prepared to pay them more money and probably not ask them to work quite so &apos;ard.\nWe know immediately whether they&apos;re a genuine possibility if the announcer on SSN prefaces the player&apos;s name with &apos;injury prone&apos;, &apos;addict&apos;, &apos;convicted murderer&apos;, &apos;cat-strangler&apos;, &apos;pensioner&apos;, &apos;free agent&apos;, &apos;recently retired&apos;, &apos;unknown&apos; and so on.\n3. &apos;ard working players from the championship who are not quite good enough but might be as long as "they buy into this football club&apos;s &apos;ard work ethic of &apos;ard work and then more &apos;ard work followed by a day of really really &apos;ard work as a reward for all their &apos;ard work "\nOur only signing to date, Jonathan Woodgate, is the living (or barely living) embodiment of Category 2, suffering, as he does, from every single complaint listed above.\nBut apart from him, we remain seated in front of SSN with no expectations other than to be informed of a whole host of players who&apos;ve turned us down for being unglamorous, not in the north and not in the south (that’ll be the midlands then), not paying enough, not liking our style of football, not liking cold, wind or rain, having an allergy to the angle of the roof on the Family Stand, their agent not liking the M6, there being no Pizza Express in Stoke, and preferring Cliff Richard to Tom Jones thus only being prepared to sign if the fans dispense with Delilah and adopt Wired For Sound as their anthem.\nWe are the masters of the last minute deal, many of whom we get really, really excited about like Eidur Gudjohnnsen only to discover the reason they came to us was.... was..... well, actually none of us can work out why Eidur came to us except that we were pathetically grateful to sign a player of his stature and prepared to pay him a wad of cash without realising he&apos;d formally retired from the game but singularly failed to inform anyone of the fact.\nAnd then there are the last minute signings that barely warrant a raising of the head from the summer long transfer inactivity torpor, like Jon Walters & Dean Whitehead, who turn out to be worth a thousand Eidur Fatjohnnssens and exemplify further TP&apos;s mastery of the dark arts of footballing alchemy.\nWe sit, we wait, we arch an eyebrow, we groan, we nod off, and we repeatedly utter the mantra that has served us so well in the past: &apos;Trust in Tony&apos;.\nOr we pretend to support Sunderland for the close season.'
            #     },
            #     {
            #         "text": 'SAN DIEGO -- The San Diego International Auto Show is scheduled to open a four-day run at the convention center Thursday, with 400 new-model vehicles, alternative fuel cars, exotics, crossovers and classics on display.\nVehicles from 36 manufacturers will be shown, including, among others, a 2017 model year Porsche 911 Carrera coupe with new twin turbo engines, and a newly designed Lincoln Continental.\n"The cars continue to be the stars," said Kevin Leap, show director. "Today\'s vehicles shine more brightly than ever with a level of quality, design brilliance, and tech savvy that has never been seen before."\nAttendees will be able to test drive cars to experience various features first-hand, check out environmentally friendly vehicles and enjoy entertainment, according to the New Car Dealers Association of San Diego County, which organizes the event.\nThe show floor opens daily at 10 a.m. Closing times are 6 p.m. today, 9 p.m. Friday and Saturday, and 7 p.m. Sunday.\nAdmission for attendees 13 over is $12. The cost is $9 for seniors 62 and over and military with identification, and $8 for youth 7-12 years old. Children 6 and under are free when accompanied by an adult.\nChildren 12 and under are free on Ford Family Day on Sunday, when accompanied by a paying adult.\nParking is available for $15 at the San Diego Convention Center. Show organizers, however, encourage visitors to take the trolley to the Gaslamp stop.\nInformation on discount coupons or VIP e-tickets is available at SDAutoShow.com .'
            #     },
            #     {
            #         "text": 'The new head of the Boston FBI Field Office sat down with NBC10 Boston to talk about his new role.\nHe\'s the FBI\'s new man in charge in Boston: Joseph Bonavolonta, special agent in charge of the Boston field office, sat down with NBC10 Boston Monday to talk about the current threats the bureau is tackling.\n"What keeps me up at night would simply be, \'What is it we don\'t know or what we aren\'t aware of,\'" said Bonavolonta.\nHe said the biggest threats are violent crime, gang violence, terrorism and cyber attacks.\n"We have nation states that are also backing or sponsoring criminal actor to engage in a wide variety of cyber-crimes," said Bonavolonta.\nThe FBI veteran took over the field office in Chelsea in January. He now oversees several high profile cases, including "Operation Varsity Blues."\nIn March, federal investigators announced the arrest of 50 parents, coaches and high profile celebrities in what\'s been called the biggest college admissions scandal in history.\n"We believe all of them parents, coaches and facilitators lied, cheated and covered up their crimes at the expense of hardworking students and taxpayers everywhere," said Bonavolonta at the March 12 press conference.\nIt\'s a case that remains active.\n"As you know, that is an ongoing and active investigation, so I\'m not going to comment any further than what we\'ve already stated based on the press conference subsequent to the arrests in that case," said Bonavolonta. Asked if was still an ongoing and evolving case, he said, "Yes."\nBonavolonta took over the post from Hank Shaw, but he\'s no stranger to this field office. He served as assistant special agent in charge from 2013 to 2017. His father was also in the bureau for 24 years and worked on organized crime investigations in New York.\n"You could say the FBI is in my DNA. It\'s in my blood," said Bonavolonta.\nThe threats are always changing. Right now, the Boston field office is heavily involved in security preparations for the upcoming Boston Marathon.\n"We are incredibly focused on determining if there is any type of intelligence that could lend itself toward a credible threat," he said. "As we sit here right now, we have not determined any."\nSix years after the attack at the finish line, terrorism, both foreign and homegrown, remains a top threat.\n"I think now, when you look at what one of our primary focuses is within counter-terrorism program, it\'s home-grown violent extremists," said Bonavolonta.\nHe added that the bureau continues to work around the clock on marathon security.'
            #     },
            #     {
            #         "text": "Politico, which broke the PR story, reports, “The group circulated a memo to reporters and producers late Monday that discouraged them from airing the undercover videos, arguing that they were obtained under false identification and violated patient privacy. ‘Those patients’ privacy should not be further violated by having this footage shared by the media,‘ the memo reads.” Patients’ privacy? What about the victims whose body parts are sold? Planned Parenthood technicians may find that kind of depravity chuckle worthy, but congressional Republicans don’t, which is why Senate Majority Leader Mitch McConnell will hold a vote before August recess to end the more than $500 million Planned Parenthood extracts annually from taxpayers. Amazingly, Minority Leader Harry Reid responded, “Good luck with that. We’re dealing with the health of American women, and they’re dealing with some right-wing crazy.” With even more videos set to be released, it’s only a matter of time before the story goes mainstream. Let’s see who the public will think is crazy — those who want to protect the sanctity of life, or those who try to justify the trafficking of human organs."
            #     },
            #     {
            #         "text": "Gaza is an open air prison where one and a half million people are deprived of adequate water, electricity, opportunities to farm, and the right to move about their own land or the world freely. Palestinian children are detained without charge for months and when charged, they are tried in military courts. Gazan schools are insufficient and the funds often cut. An Israeli museum is built over a centuries old Muslim graveyard. Ancient olive trees, the livelihood of Palestinian farmers are burned. Homes are bulldozed. A blockade prevents goods from coming into and out of Palestine, inhibiting commerce and the means for a livelihood. Water is diverted from its source directly to Israel where it is abundantly available, including for swimming pools, while Palestinians have to buy water from tanker trucks. (A Durham Presbyterian church funded water purifiers for Gaza schools after the bombing of U.N. schools.) The wall closing off Israel from Palestinians makes everyday life difficult. Israeli settlements take more land from Palestinians every year. Drones fly 24 hours a day spying on and frightening the residents of Gaza.\nThe asymmetry of the two sides is extreme. A few rockets a week fall into north Israel and if they hit an Israeli citizen, Israel attacks with an over-the-top force, killing hundreds of Palestinians. Palestinians have only stones, and now knives, to express their frustration and anger.\nThose who support Israel based on religious beliefs do not realize that Palestine Christians also suffer under the burden of the Israeli occupation. The Presbyterian Church of America, and The Episcopal Peace Fellowship have spoken out about the repression in their book and DVD called, “Steadfast Hope” for their parishioners. It is painful to watch and read.\nPalestinian student campus organizations face bureaucratic harassment, severe restrictions on and censorship of their columns, even cancellations of their speakers. If they do win the battle to publish their viewpoint, Jewish organizations immediately spring into action to overwhelmingly deny and refute their message.\nMovies, media and games perpetuate stereotypes with Palestinians depicted as terrorists and villains, vicious gunmen, wide-eyed maniacs killing anyone, anywhere, any time for any reason. Movies and television programs such as “Tyrant,” “Dig” and “Homeland” depict Arabs as the lowest of human beings.\nThis in contrast to positive portrayals of Israelis, such as Ziva David as a Mossad agent in “NCIS.” Europeans are speaking out about the Gaza tragedy, but American voices are silenced out of fear of being accused as anti-Semitic. A major conference, “The Israel Lobby: Is It Good for the US, Is It Good for Israel?” received no main stream media coverage.\nHow to get the message out? Jewish organizations inside Israel and the U.S. are working for peace and to let the world know of the suffering of the Palestinians. That is the message the protesters want to bring to the council and the public.\nEleanor Kinnaird is a former state senator and mayor of Carrboro."
            #     },
            #     {
            #         "text": "This mildly racy spot introduces us to the character of the brown M&M, who only looks like she’s not wearing her shell. And it does so by showing us more than we ever cared to see of one of her bagmates. “That is not something I want to be thinking about when I’m eating M&M’s,” my 10-year-old son says. He’s got a point, but be thankful: if someone bared his peanut, we’d have an FCC issue right now."
            #     },
            #     {
            #         "text": "This week in Washington, Cliff Floyd is expecting the best – but bracing for the worst.\n“I’m expecting good, clean baseball,” said Floyd, who was 3-for-18 with six strikeouts against Dontrelle Willis and was given yesterday off. “I think if Pedro [Martinez] hits [Jose Guillen], we’re going to fight.\nFloyd has never understood that when a pitcher hits a position player, a position player on another team is targeted.\nHe said he’d be seething if his team had been hit six times in a series, as the Nationals were.\n“But my point over there is, hit the guy who’s throwing,” Floyd said. “He’s going to pitch again, sooner or later.\nThe Mets left fielder nearly charged the mound against Houston’s Roy Oswalt last season, and he acknowledged talking about retaliating (as he did at the time) is nonsense that must stop.\nBatting .200 (3-for-15) this season, Floyd worked on his mechanics in the indoor cage before the 3-2 victory over Florida.\nThe Mets realigned their rotation after Saturday’s rainout, and Victor Zambrano will start Thursday in Washington. Zambrano (strained left hamstring) was originally scheduled to start yesterday.\nBrian Bannister and Pedro Martinez will stay on turn and pitch Tuesday and Wednesday. Tom Glavine and Steve Trachsel will flip-flop and work on Friday and Saturday so Glavine can stay on turn.\nCarlos Delgado was inconspicuous during the seventh-inning rendition of “God Bless America,” when a handful of his teammates stood on the top step or above it. . . . Saturday’s rain washed away starts for Chris Woodward and Ramon Castro. Yesterday, Paul Lo Duca and Anderson Hernandez were back at catcher and second base, respectively. “We’ve got an off-day [today], and I didn’t want guys playing every day to get three days off,” Willie Randolph said. “That’s just the way it works out.” The Mets play day games after night games on Thursday and Saturday, so Castro should start a couple of games. Woodward pinch-hit in the sixth and whiffed against Willis."
            #     },
            #     {
            #         "text": "James finished the preseason ranked tied for 14th in scoring (13.7 points per game) among league forwards.\nExhibition numbers mean little, or perhaps nothing, in the grand scheme of a season, but Bosh’s statistical effort during the preseason is anything but irrelevant — or random — when taken in the correct context. Heat coach Erik Spoelstra spent the entire training camp drilling his players on the idea of a free-flowing, equal-opportunity offense. That Bosh, Mr. Random Guy, emerged from training camp as the team’s leading scorer means the Heat’s offense is healthy heading into the season opener against the Celtics on Tuesday at AmericanAirlines Arena.\nMore often than not, Bosh is open on the offensive end. The ball found him during the preseason, and Bosh’s skills did the rest. He led the Heat in field-goal percentage (.551) while also leading the team in rebounds (43).\nJames is the driving force behind Spoelstra’s idea of “positionless basketball,” but Bosh might be the second-most important cog in the wheel. On most nights, Bosh is the quickest big man on the floor, and his consistent jump shot allows him to stretch defenses.\nConventional defensive strategy against the Heat calls for packing the paint to account for Wade and James. Entering his third season with the Heat, Bosh knows where to find the open spots on the periphery. He likes to call those opportunities random, but, when paired with James, the results are more like basketball jazz.\n“He’s open,” said James, who led the Heat with 41 assists in the preseason. “We’re playing with a lot more pace and a lot more space for him, and giving him a lot more opportunities to go at his matchup.\nWade credits Bosh’s big preseason to his work before training camp. Bosh skipped the Olympics to fully heal from the abdominal injury he sustained in the playoffs. Upon his return, he met with Wade and James and vowed to carry more of the load early in the season.\nWhen Bosh first arrived in Miami, the Heat wanted him to pack on muscle mass and bulk up for a life in the paint. Spoelstra scrapped that plan after the playoffs when the Heat used an unconventional lineup to win its second NBA championship. The Heat’s coach instructed Bosh to focus on his natural skill set: speed, quickness and shooting.\nBosh averaged 18.0 points per game last season, but Wade says Bosh is capable of “going for 20-plus a game and probably more effortlessly than” himself and James.\nWhile Bosh has accepted the label of center, offensively he’s more of a hybrid power forward than anything. The position of a traditional center is a dying art in the NBA, so much so that the league has removed the designation from All-Star ballots this season.\nFans will now vote for three “frontcourt players” and two guards rather than two guards, two forwards and a center. Despite the change, Bosh still doesn’t expect to garner any more votes for a starting spot.\n• The Heat finalized its 15-man roster on Saturday, cutting point guard Garrett Temple. Miami begins the season with 12 players from its championship team. Forward Rashard Lewis, forward Josh Harrellson and guard Ray Allen are the new additions."
            #     },
            #     {
            #         "text": "Even if you’re not at risk of dying, you can still get other people sick.\nThe number of people who died from the flu in the 2017-2018 season.\nEstrada Anton // Shutterstock - Even if you will not die from flu, your actions affect others.\nIt feels like up until a couple years ago, the accepted line about the flu shot was that you only needed it if you were a) young, b) old, or c) sick, and that maybe it didn’t work that well anyway because it only protects against certain (the most common) strains. Millennials received this info gratefully; finally, a thing we were not responsible for, an errand we did not actually have to do. Unfortunately, this is wrong; in fact, everyone should get the flu shot.\nLast year’s flu season was the worst in a decade, worse than the year of swine flu. Over 80,000 people died. There are many factors at work, but a big one that medical professionals attributed to the unusually high rate of deaths and infections was a drop in the rate of adults who bothered to get their shot — yes, those same people between the ages of 18 and 65 who “don’t need” it.\nTo think about vaccines as they affect each person individually is blinkered; sure, you don’t want to get sick, but more than vaccines prevent individual illness, they prevent illnesses from spreading. We see this already with children in locations where it’s trendy among parents to simply not get their kids vaccinated from preventable diseases like measles out of irrational fears; because of those parents, those diseases spread faster and people die more frequently. Epidemics happen because of new, wild, aggressive disease strains, but they also happen for lack of prevention. More to the point, just because you can afford to miss work or buy medication when you get sick doesn’t mean others can.\nIn the same way you don’t not vote because nothing is bothering you specifically, you don’t not get the flu shot because you are very likely to survive it yourself. This is how social contracts work: How a collective action impacts you personally is maybe the least important thing, especially if you are in the most privileged group. Please get your flu shot."
            #     },
            #     {
            #         "text": 'Nelson Mandela statue in Westminster, London.\n"Comrade Mandela\'s release was achieved through our struggles that had pushed the apartheid regime into a corner where they were forced to negotiate their way out of power," he said.\n"Mandela\'s freedom from his prison cell was also brought about because of unrelenting pressure by our international allies."\n"We are eternally grateful that he had lived to enter a new country. Here in the Western Cape we pledge to continue Mandela\'s fight for a just society."'
            #     },
            #     {
            #         "text": 'Is borrower required to pay?\nDEAR BENNY: We are in the final steps of completing a refinance of our barely year-old $410,000 mortgage. We were pleased with the interest-rate drop, and our local bank was generous in dropping many of the so-called "junk fees" associated with a refinance. However, we are being charged $1,007 for title insurance. When I asked our banker about this, the response was basically, "Well, yes, it is a rip-off but there is nothing we can do about it."'
            #     },
            #     {
            #         "text": "Well maintained home situated in private culdesac. Home boasts lots of light, 4 bedrooms + office, large backyard perfect for entertaining, shed and fireplace with mantel. Brand new carpet on main, newer water heater, furnace with ionizing air filter for better air quality.\nGently used, clean, split entry home, with granite counter tops and laminate vinyl flooring! Close to Bangerter Highway, and Jordan Landing. No backyard neighbors as home backs up to park. Quiet neighborhood, but central to everything.\nVIEWS VIEWS VIEWS. Check out the tour. Sits high on the west bench. Awesome view from the front porch and master bedroom window. Very quiet neighborhood, no through traffic. Fully finished basement, all new carpet throughout, new granite kitchen."
            #     },
            #     {
            #         "text": "councilors at the USC Annenberg School for Communication.\nArkansas governor and congressman who joined the board in 2006.\nrenewed attention to noncommercial broadcasting.\nThe elections came at CPB&apos;s board meeting in New Orleans Tuesday (Nov.\n16). Ramer succeeds Ernie Wison as chairman.\nThe newly elected chairman of the Corporation for Public Broadcasting talks with B&C’s John Eggerton about transforming noncommercial broadcasting into multiplatform, locally-focused public-service media."
            #     },
            #     {
            #         "text": "Latest in \"Revolution in Rojava\"\nWho are they, these revolutionary Rojava women?\nMeredith Tax just had to find out who they were - the revolutionary women of Rojava, bearing arms against ISIS, building a new world...she had to find their story, for herself, and in her new book, for us.\nRojava revolution: how deep is the change?\nIs optimism in the future of revolutionary change misplaced in a region torn apart by war and a society where patriarchy has been so entrenched? Part 6 of Witnessing the Rojava revolution.\nRojava is a fast moving, dynamic place where things change by the minute. What are the material conditions which support this woman-centred revolution ? Part 5 of 50.50's series Witnessing the revolution in Rojava, northern Syria.\nRojava's battle with ISIS stronghold Raqqa is not simply a military one, but an ideological one in which the position of women could not be more polarised. Part 4.\nIn less than four years, the women’s umbrella organisation, Kongira Star, has set up an autonomous, grassroots, democratic structure which has resulted in shifting patriarchal mindsets and reversing gender discriminatory laws. Part 3.\nTravelling in Rojava is to witness the ways in which the different commitments to the revolution present a conundrum. How can one system satisfy the vast differences in human aspirations? Part 2. Part 1.\nTravelling in Rojava is to witness a revolution experimenting with a form of stateless, direct democracy with women’s liberation, race and class equality at the heart of it. Part 1."
            #     },
            #     {
            #         "text": "It may be noted that Air India's was the only service that plied between Madurai and Mumbai.\nSome passengers pointed out that all the tickets used to be sold out during the festive season and summer.\nChennai: Much to the chagrin of air passengers and traders of Madurai, Air India announced the temporary cancellation of its Mumbai to Madurai flight via Chennai from July 12 to July 31, citing low patronage to the service that has been catering to passengers of the southern districts for almost 42 years.\nAlthough the announcement of cancelling AI671 (Mumbai to Madurai) and AI672 (Madurai to Mumbai) said that the cancellation has been done on a temporary basis, Air India sources said that the operation of the service from August 1 would be decided in the coming days. AI671 was taking off from Mumbai and reaching Chennai at 10.45 am every day from 1976.\nThe flight would leave for Madurai at 11.30 am. It was taking off from Madurai to reach Chennai at 2.15 pm. The decision was taken at a time when local passengers are demanding a direct service from Madurai to Mumbai.\n“Private airlines are operating as many as 16 services between Chennai and Madurai. But Air India claims that the flight has low patronage. How can a private player operate flights without patronage? Air India is making way for private airlines,” a regular passenger alleged.\nIt may be noted that Air India's was the only service that plied between Madurai and Mumbai. “Many traders were using the Air India flight to send their jasmines, vegetables and other agricultural products to Mumbai for years. Withdrawing the flight service will hurt traders and farmers of the southern districts,” the passenger added.\nSome passengers pointed out that all the tickets used to be sold out during the festive season and summer. “On normal days too, nearly 80 percent tickets were sold for the Air India flight. The claims of poor patronage made by the authorities are misleading,” they alleged.\nMeanwhile, union minister Pon. Radhakrishnan said that he would talk to the Civil Aviation Minister to operate the existing flight service and an additional flight between Madurai and Mumbai."
            #     },
            #     {
            #         "text": "EXCLUSIVE: Actor and comedian Joel McHale has signed with UTA and Anonymous Content for representation. Previously with WME, McHale is perhaps best known for starring as Jeff Winger for six seasons on the NBC/Yahoo comedy Community and spent twelve seasons as the host of E!’s satirical series The Soup. Most recently, he toplined CBS’s short-lived sitcom The Great Indoors.\nMcHale will next be seen in the Netflix biopic A Futile & Stupid Gesture, portraying former Community co-star Chevy Chase, and in Sam Levinson’s upcoming thriller Assassination Nation.\nOther credits include A Merry Friggin’ Christmas, where he played Robin Williams’ son, the Jerry Bruckheimer-produced supernatural thriller Deliver Us From Evil, Warner Bros’ rom-com Blended, and Seth MacFarlane’s blockbuster comedy Ted."
            #     },
            #     {
            #         "text": 'Jesse Winker will undergo season-ending shoulder surgery. He was batting .299 this year with 7 homers and 43 RBI.\nThroughout the last couple of seasons, Jesse Winker dealt with pain his right shoulder.\nAfter he felt more pain Monday, it was determined that Winker would need surgery to repair his shoulder. He will miss the remainder of the season.\nThe Reds placed Winker on the 10-day disabled list Thursday with a right shoulder subluxation.\n"Man, I have no idea it is," said Winker. "I know my shoulder hurts. That’s all I know."\nWinker said the pain was "off and on" all season but Monday was essentially the final straw.\nOn first base during a comeback win over the St. Louis Cardinals, Winker ducked under a line-drive single from Tucker Barnhart in the ninth inning and fell to the ground on the base paths. Winker later scored the game-winning run.\nA potential Rookie of the Year candidate, Winker was batting .299 this season with seven homers and 43 RBI. The outfielder ranked fourth in the Majors with a .405 on-base percentage. He had more walks (49) than strikeouts (46).\nSpeaking to reporters Thursday, Winker was emotional about the end of his season.\n"Obviously, I was hoping that surgery wasn’t the end result but I’ve been dealing with this for 2-3 years," Winker said. "Just got to the point now where it’s time to go fix it."\nIt\'s a tough loss for the Reds\' lineup, which was already without outfielder Scott Schebler. On the DL, Schebler began a rehab assignment Wednesday, going 1-for-3 with a walk as a designated hitter at Triple-A Louisville.\nWinker, based on what he\'s been told from doctors, said his goal is to return by Feb. 1, 2019. He said he felt the pain when he completed "any baseball move."\n"There were times where it affected his swing," Reds interim manager Jim Riggleman said. "He altered his swing path because of the pain. That’s a tough way to play at the Major League level. He did a heck of a job. I don’t think anybody realized that he was going through it as much as he was."\nMason Williams, an outfielder at Louisville, was called up to fill Winker’s roster spot. Williams played in 25 games with the New York Yankees over the past three seasons.\nA left-handed hitter, Williams signed with the Reds as a minor league free agent last offseason.\nWilliams was batting .280 with six homers and 30 RBI in 87 games at Louisville. He recorded a hit in 13 of his last 14 games.\n"It’s been a matter of me being healthy and staying on the field and playing my game and just having confidence in myself," Williams said.\nWilliams, once considered the Yankees\' top prospect, learned of his promotion to the big leagues Thursday morning.\nHe played on the same high school team in Florida as Cardinals pitcher Austin Gomber, who had a no-hitter against the Reds for six innings Tuesday. The two players will sometimes train together in the offseason.\nTo make room on the 40-man roster for Williams, the Reds moved injured infielder Alex Blandino (torn ACL in his right knee) to the 60-day DL.\n"He’s had productive at-bats, competitive at-bats, against both left-and-right-handed pitching," Riggleman said of Williams. "He’s a good athlete. We’re confident he will come up here and do a good job."\nSCHEBLER STILL PROGRESSING: Schebler was in Louisville\'s lineup as a designated hitter for his second rehab game Thursday.\nBut the key to his return will be when he\'s comfortable making throws.'
            #     },
            #     {
            #         "text": "Office space has recently been renovated and is ready for occupancy. Located in the heart of downtown the space is convenient to the Post Office and local convenience store. There is plenty of off street parking with rear entrance available.\nThis ranch offers 3 Bedrooms, 2 full baths. Enjoy the back deck for grilling & chilling on those summer days. Call today!\nMaticulously mantained 3 Bedroom home with a bonus family room on the first floor, and finished attic. 1st floor open floor plan with updated kitchen and Stainless steel appliance. Laundry room conveniently located off the kitchen."
            #     },
            #     {
            #         "text": "An exploration of the fundamental drivers behind long term shifts in the demand for, and supply of, land for agriculture, forestry and environmental uses over the next four decades. Topics include trends in food and bioenergy demand, crop productivity on existing and potential croplands, water and climate constraints, non-extractive uses such as carbon sequestration, and the role of global trade and public policies. Students will lead discussions of weekly readings and perform simple numerical experiments to explore the role of individual drivers of long run global land use."
            #     },
            #     {
            #         "text": 'Question: What\'s one company culture characteristic that you have found makes your startup employees the happiest? How do you make sure you\'re implementing it?\n"Every six weeks, we have scheduled, highly structured bi-directional reviews with every employee. The predictability and structure of these meetings make it extremely easy for employees to deliver feedback, both good and bad. This level of transparency and communication keeps employees happy and motivated."\n"At RTC, we have a motto as a publisher: "Don\'t expect your reader to change through reading your book if you haven\'t changed through writing it." That motto defines our core belief that continual growth is necessary for the human spirit to regularly experience joy in the workplace. As a result, our executive team strives to help our staff remove aspects of their work that do not bring them joy so that they can focus on what they love doing. By intentionally making room for them to focus on what they enjoy, they are able to grow out aspects of the business that bring them deep personal satisfaction while also serving our clients. We\'ve developed entire new lines of business this way, as well as new positions within the company. Support their dreams, and they will grow your business."\nFollow Rule #2: Have Fun!\n"When my co-founder and I started \'ZinePak, our business plan read simply, "Make money. Have fun." As the company has grown, we\'ve made sure not to lose sight of this mission. At the end of each day, we ask our employees if they had fun that day. Almost without fail, the answer is always "yes." We try to always remember that we\'re an entertainment company. We aren\'t changing the world. We aren\'t curing cancer. No one\'s life is at risk, so there is no need for the doom-and-gloom culture that seems so prevalent in Corporate America. From half-day Fridays to candy jars to days off for charitable activities of the employee\'s choice, we try to mix "fun" into everything possible to ensure that work feels as much like play as possible."\n"We always take time to celebrate our wins. Whether it\'s a new project, new teammate or new launch, we take time to recognize team and individual successes. Taking small breaks to socialize and catch up at team happy hours reenergizes the team and ultimately leads to awesome workflow and collaboration. Our internal party planning committee makes consistent plans to pull us away from our desks and into fun environments where we can take our minds off work for a bit."\n"I think that people are generally happiest at work when they are engaged by the work that they do. Doing the same jobs, having the same responsibilities and facing the same tasks day in and day out can get tiresome. In my company, there is a variety of work to be done, and employees are encouraged to embrace the variety. This keeps work fresh. We also support telecommuting (to the degree that we don\'t even have a "home" office -- all work is done "off-site") and flexible schedules. We trust our team to do their work when they can and where they can. Work variety and flexible work results in employees who are happy and productive. "\n"When we started, we had been told to be careful of what we share with our employees and other stakeholders. We, however, are very open and transparent communicators and did not keep anything confidential and didn\'t hold back any information. This allowed our early employees to not only feel like they were playing an important trusted role in making an impact on growing the company, but also allowed them to dissent and suggest better ways of accomplishing objectives. They respected the founders because they saw not just what decisions we made, but how we made them, right or wrong. And for failed experiments, we had their support and morale to pivot quickly. Each and everyone felt individual ownership for each decision and worked that much harder to succeed, because they never felt separate."\n"Every other Friday, someone different leads our team workout. We\'ve played tennis and basketball, done yoga and CrossFit and even learned (barely) various martial arts. Unintentionally, we\'ve taken risks doing new things, discussed how we\'re improving our lives (not just our work) and laughed a lot. The benefits of exercise combined with the additional vulnerability, camaraderie and fun has increased the happiness quotient of Team Fig. "\n"When people hear the words "company culture," they often think about ping pong tables or beer taps. While those sorts of perks are cool, they really don\'t matter unless you\'ve created the right work environment to embrace them. You create that environment by giving people a voice. When we built our new corporate office space, it was very important to solicit opinions and ideas from our employees. Before moving into our new facility in 2011, we hosted an internal version of Pinterest where employees could put ideas and pictures that they felt should be considered for the new space and workstation setups. In the end, when you\'re making a decision based on democratic feedback, you need transparency. People will be invested in the outcome as long as they feel like the process is fair. "\n"The characteristic that I swear by is living the mission every day. Oftentimes, people join your team/company as a startup because you are doing something different or you\'re doing something the way no one has done it before. You cannot afford to lose that, and you have to live that everyday. For us, that is our mission. Everyone who has joined our team is in it for the mission, and we push it and live it every day, which makes our team members happy. "'
            #     },
            #     {
            #         "text": 'Art can be found anywhere. And it doesn\'t have to be created with paint on canvas, sculpted from clay, or chiseled in stone.\nSometimes all it takes is a simple piece of chalk and a public sidewalk. That\'s the theory behind the annual Chalk Art Festival, which is June 13 at the Uptown Shopping Center in Richland. Registration starts at 8 a.m. Chalkers will begin their work at 9 a.m. and continue until 3 p.m.\n"Chalk art is a wonderful and unique form of creativity," said Gus Sako, the event\'s organizer and owner of the Octopus\' Garden novelty store in the Uptown.\n"Give the smallest toddler a piece of chalk and sidewalk and they will be happy for hours," he said.\n"I get off work and usually just head to the sidewalk," she said with a chuckle. "It\'s just too irresistible not to take part."\nBoth women also say creating chalk art can wreak havoc on the body, especially the knees, the back and shoulders.\n"It\'s pretty hard for anyone to whip out a drawing in the hot sun on a dirty sidewalk. And, at almost 66, my knees are pretty creaky," Loomis said. "On the other hand, my kids just gave me a gardening stool with rails that might be just the thing!"\nCalicoat says creating chalk art is like doing one-arm push ups for hours.\n"I\'ve been trying to train myself to use my left hand to paint so I\'ll probably do the same thing with chalk," Calicoat said.\nThe Chalk Art Festival was started by the now defunct Corporate Council for the Arts as a regional arts activity, Sako said. The festival moved to several locations until it finally settled at the Uptown a few years ago.\nCategories and registration fees are: Up to age 5 $5; Ages 6-9 $5; Ages 10-12 $7; Ages 13-17 $7; Ages 18 and older $12. Sidewalks to be used for the chalk artistry will be on George Washington Way, Jadwin Avenue, Symons Streets and Williams Boulevard.\nFor more information, call 946-0077 or 943-6542.'
            #     },
            #     {
            #         "text": "https://shop.bbc.com/guinness-harp-baseball-cap-21685.html?___store=en_us 4964 Guinness Harp Baseball Cap https://shop.bbc.com/media/catalog/product/2/1/21685-guinness-baseball-cap_1_.png 21.98 21.9800 USD InStock /New to the Shop /Apparel /Gifts /Apparel/Accessories /Fall Catalog /Holiday Catalog /Shop by Price/Gifts Under $25 When the strings of a Gaelic harp design appear on the left, it’s the official emblem of Ireland. When they’re on the right, it’s the classic trademark of Dublin’s most famous brewery, established in 1759. When the harp is embroidered in golden thread and appliquéd on a grey cotton cap, you’re ready for a good day, wherever you are. Sturdy bill with top-stitching shades your face from the sun.\nBack strap with golden, embossed clip adjusts for a perfect fit. Official Guinness merchandise. 100% cotton.\nWhen the strings of a Gaelic harp design appear on the left, it’s the official emblem of Ireland. When they’re on the right, it’s the classic trademark of Dublin’s most famous brewery, established in 1759. When the harp is embroidered in golden thread and appliquéd on a grey cotton cap, you’re ready for a good day, wherever you are. Sturdy bill with top-stitching shades your face from the sun."
            #     },
            #     {
            #         "text": 'Up to an inch of snow is likely on the Eastern Shore and up into Sussex County Friday afternoon.\nBelow-freezing temperatures this week is leading to extra slippery roads Friday as snow falls across the Delmarva Peninsula.\nMultiple school districts and Sussex County government offices have closed early for the afternoon snow that is supposed to linger into the early evening, according to Eswar Iyer, a meteorologist with the National Weather Service in Wakefield, Virginia.\n"The heaviest snow should move east in the next couple hours and might not taper off until later afternoon and early evening," Iyer said just after 2 p.m.\nAccumulation totals have not changed from the morning prediction of 1 inch, but some areas could see a little more, Iyer said.\nMultiple vehicle crashes have been reported on the police scanner as roads have become icy, which is due to the cold temperatures leading up to Friday.\n"When it’s colder, the snow is able to stick to the roads quicker, and your roads are going to get slick quicker," Iyer said.\nThe Ocean Police Police Department announced at 2:40 p.m. that westbound Route 90 lanes were closed because of a crash.\nIt could take up to two hours to clear the crash. Maryland State Police is investigating.\nRt. 90 is closed between Rt. 589 and Ocean Due to poor road conditions. Avoid the area.\nSalisbury Police tweeted they have nine active crashes under investigation as at around 3 p.m.\n"Snow catches us by surprise today and causes very icy road conditions," police said.\n"Slow down, use caution if you must be on the roadways this afternoon."\nTemperatures were forecasted to climb close to the freezing point for water in Salisbury, while Georgetown and Rehoboth Beach were only going to reach 28 degrees. It will be a little warmer in Ocean City and Accomac, with temperatures in the mid to upper 30s.\nThe temperature won\'t drop significantly at night, with a low of 20 degrees in Salisbury, 18 degrees in Georgetown and 26 degrees in Accomac.\nOver the weekend, expect temperatures to rise into the 40s across Delmarva. By Monday or Tuesday, Delmarva residents can expect temperatures in the low 50s.'
            #     },
            #     {
            #         "text": '10 diversity items for June 29: Unemployment up in most U.S. cities; Pew report shows diversity of U.S. Hispanics and more.\nFor the first time in history the Pentagon celebrated Lesbian, Gay, Bisexual, Transgender Pride Month on Tuesday, the Los Angeles Times reported. The ceremony, which was broadcast on a internal TV network to U.S. military bases around the world, was a straight-laced affair, according to the Times. It included pre-taped videos from President Obama and Defense Secretary Leon E. Panetta.\nThe American Civil Liberties Union will help the Ku Klux Klan in its bid to join a highway cleanup program, according to Fox News. When the International Keystone Knights of the KKK applied to join the program along part of Highway 515 in the north Georgia mountains, the state denied their application--which lead to a legal showdown. The ACLU is developing a strategy for representing the group in what it believes is a First Amendment case.\nWhen advocates for the Asian-American community decried a report by the Pew Research Center full of seemingly good news about Asians as "shallow"\x9d and "disparaging,"\x9d both sides failed to acknowledge that the other may have had a point, Eric Liu wrote in Time on Tuesday.\nIn May, unemployment rates rose in more than 75 percent of U.S. cities, the Associated Press reported on Wednesday. Among the cities with this highest unemployment rates were Yuma, Ariz. (28.9 percent); El Centro, Calif. (26.8 percent); and Yuba City, Calif. (17.9 percent). Bismark and Fargo, both in North Dakota, had the lowest unemployment rates--2.5 percent and 3 percent respectively--followed by Lincoln, Neb., with 3.4 percent unemployment.\nEighty percent of Mexicans support their president\'s decision to use the Army to fight powerful drug cartels, a new poll from the Pew Hispanic Center shows. That support has dropped slightly over the past year. In 2011, 83 percent supported the use of military force. Forty-seven percent of those polled said they believed the Army was making progress in the fight.\nMexicans, Puerto Ricans, Cubans, Salvadorans, Dominicans, Guatemalans, Colombians, Hondurans, Ecuadorians and Peruvians make up 92 percent of the United States\' Hispanic population, according to a Pew Hispanic Center analysis of Census data released on Wednesday. The majority, 65 percent, of all 50.7 million Hispanics living in the country are of Mexican-origin. The next largest group are Puerto Ricans, who make up just 9 percent of the total Hispanic population.\nThe attorney for a Latino man who claimed a Seattle police officer threatened to beat the "Mexican piss"\x9d out of him during a 2010 robbery investigation said a civil rights lawsuit regarding the incident, which was caught on tape, has been settled for $150,000, the Seattle Times reported on Wednesday.\nTucson Police Chief: Can the Department Handle S.B. 1070 Workload?\nAs the "show me your papers, please"\x9d provision in Arizona\'s immigration law goes into effect in the wake of the Monday Supreme Court decision, the police chief in Tucson, Ariz., wonders how his staff--which is down to 160 officers because of the economy--will handle the up to 50,000 additional phone calls a year to federal officials to verify the immigration status of people they stop, CNN reported on Wednesday.\nThe U.S. government has quietly been training and arming the Ugandan military as it drives militants out of Somalia, a stronghold for Islamic militants, Wired reported on Wednesday. But American officials have indicated that the government might cut off that military aid because of LGBT issues. Uganda\'s gays, lesbians and transgendered citizens have long faced persecution, the magazine reported.\nAnother lawsuit challenging Florida\'s contentious move to remove potentially ineligible voters from the state voting rolls was filed last week by The Advancement Project in partnership with other litigants, New America Media reported on Thursday.'
            #     },
            #     {
            #         "text": "He and a neighbor managed to lure the horse toward them with a handful of grass, Sickinger said, and they even petted the horse for a bit. Sickinger had a strap he planned to use to bridle the horse, but he said better judgment took over and he decided the horse was too wild to control even if he could wrangle it.\nSickinger told a St. Clair County sheriff’s deputy the horse might belong to a neighbor. There are several horse farms in the area, but none came forward as the owner, Sickinger said.\nMoments after Sickinger started petting the horse, a truck drove by and the horse followed after it. No one would see the horse until more than a week later.\nOn Wednesday, area resident Aranza Lee spotted the horse in a soybean field near Imbs Station and Wagner roads, less than a mile from Sickinger’s home. She captured a video of the horse running freely through the field and shared it to the Millstadt News Facebook page. By Thursday afternoon, the post had more than 400 shares, but no one had come forward as the owner.\n“I mean, how weird is that?” said Sara Yoch of Smithton, a self-described horse-lover.\nLt. Alan Haake of the St. Clair County Sheriff’s Department said the department has received several reports over the past three weeks about the missing horse, but he said no one has come forward as the owner, nor has anyone been able to catch it.\nYoch was out at the intersection of Otten and Wagner roads Thursday afternoon with a bucket of feed and a lead, looking for the horse. She said she was out in the same area Wednesday for about four hours, but didn’t have any luck.\nShe warned area residents to avoid approaching the horse if they aren’t familiar with how horses behave, and to slow down when driving through the area. Yoch said a horse standing in a road on a dark night can cause serious damage to a vehicle and hurt the driver, as well as the horse, of course.\nIt’s possible the horse was dumped, according to Stephanie Goepfert, a member of the Lincoln Trail Riders in O’Fallon. It can easily cost more than $500 a month to care for a horse, Goepfert said, and it’s possible the owner could not afford to keep it.\nOn its own, the horse could get sick or injured, Goepfert added.\n“They can survive for a period of time in the wilderness, but if they’re a domesticated animal, they’re relying on a certain diet. It can be bad for them,” Goepfert said.\nThey can survive for a period of time in the wilderness, but if they’re a domesticated animal, they’re relying on a certain diet. It can be bad for them.\nThe most recent sign of the horse was a bedded-down area next to a creek near where the horse was spotted on Tuesday, Lee said.\nThere was no sign of the horse as of Thursday afternoon, though several groups were planning to head out and search for it.\nAnyone who spots the horse can call the St. Clair County Sheriff’s Department at 618-207-4374."
            #     },
            #     {
            #         "text": "This full stratified home has a 2 level main home and fully contained suite on the walkout lower level! Great investment or revenue opportunity; or choose your unit and sell the other! Excellent quiet location on the Knoll at Silver Star Mountain Resort with the ski-way out your back door. Beautiful Monashee Mountain views! Well layed-out floor plan with a common entry and shared double garage. Both units enjoy in-floor hot water heat, private hot tubs with shower rooms, private laundry and a full ensuite with every bedroom! The main house features a large private entry/ski storage with access to the hot tub, sauna and shower room. Great room concept living area with a grand river rock gas fireplace, 2 story windows and soaring ceilings in the living room to the upper level. The kitchen has stainless appliances and casual eating bar while the dining area will accommodate your large gatherings. 3 Bedrooms and 3 baths on the upper level including the master which has mountain views and ensuite featuring a soaker tub and separate shower. Lockable owner storage off the laundry room. The lower suite presents 2 bedrooms, 2 baths and generous open concept living with gas fireplace in the living room, a large island in the kitchen with eating bar and level walkout to the hot tub."
            #     },
            #     {
            #         "text": 'Australian internet users were the big losers from today\'s NBN Co deal with Telstra, according to opposition communications spokesman Malcolm Turnbull, as it condemns them to pay high broadband prices.\nThe $11 billion deal with Telstra paved the way for accelerated rollout of the national fibre network.\nBut Turnbull said the deal served only Telstra and the Government\'s interests.\n"The deal will have damaging consequences for consumers – that is, every Australian that purchases broadband or telephony services during the next decade," Turnbull said.\n"The NBN Co corporate plan makes it clear that broadband prices will be high and stay high."\nIn addition, the sell-off of Telstra and Optus\' HFC networks would remove the only networks that could have competed with NBN Co\'s services to keep prices low, he said.\nTurnbull continued to push the Coalition\'s line that options should have been built into the Telstra deal to give NBN Co access to copper should a "future Government" decide to can FTTP in favour of an FTTN architecture.\nThe suggestion didn\'t find favour with the Greens.\n"Today we heard the opposition communications spokesperson claimed that if elected, the Coalition will leave those parts of the NBN already existing intact, but that the remainder of the network would be a hodgepodge of Fibre to the Node (FTTN), wireless and Fibre to the Premises – a flawed model which was roundly rejected in the 2010 election campaign," Greens Senator Scott Ludlam said.\n"This suggestion has nothing to do with communications reform".\nLudlam said the Opposition had been delaying the NBN for a year, "hysterically predicting doomsday scenarios for the sector."\nHe said constructive input on telecommunications reform from the Opposition would be welcome but none had eventuated.\nTelecommunications analyst Paul Budde said the deal was structured in such a way that it would be difficult for a Coalition Government, if elected, to roll it back.\n"In our eyes the future of the NBN looks now secured," Budde said. "The Opposition Shadow Minister Malcolm Turnbull has already indicated that he is not going to turn the clock back, but he of course is still planning changes if they would win the next elections.\n"It will be difficult for any government to renege on the broadband services that are now staring to emerge in the first release sites around the country, once people started to get a better understanding what this will mean for them, few people in regional or rural areas will accept a second class solution for them, simply because that is cheaper."\nOvum consulting director Nigel Pugh said that while there had "always been an overhang to the deal with regards to a change of government", the analyst firm\'s "initial reading of the cessation clauses don\'t position this deal as a poison pill if there is a change of government at the next election."\nThe Australian Information Industry Association welcomed the deal, with the proviso that it renewed the "imperative [of business] to act quickly to seize the opportunities it presents."'
            #     },
            #     {
            #         "text": "WASHINGTON: US Secretary of State Michael Pompeo said on Monday that the United States wanted a ceasefire in Afghanistan during Eidul Azha because this was also the desire of the Afghan people.\nThe Afghan government announced on Sunday that it wanted a ceasefire in the country during this Eid like the one that was observed during Eidul Fitr, which allowed rival Afghan factions, particularly the Taliban, to celebrate the religious festival peacefully with their families.\nBut Mr Pompeo and Afghan officials both said that for this ceasefire to happen, it was necessary for the Taliban to desire it as well.\n“This plan responds to the clear and continued call of the Afghan people for peace,” Mr Pompeo said.\nHe noted that the last ceasefire in Afghanistan revealed the deep desire of the Afghan people to end the conflict. “And we hope another ceasefire will move the country closer to sustainable security,” the chief US diplomat said.\nMr Pompeo said the US supported this initiative because “it is our hope and that of the international community that the Afghan people may celebrate Eidul Azha this year in peace, free from fear”. He said the US also supported Afghan President Ashraf Ghani’s offer for comprehensive negotiations with the Taliban on a mutually agreed agenda. “We remain ready to support, facilitate, and participate in direct negotiations between the Afghan government and the Taliban,” said the US diplomat.\nEarlier this week, Mr Pompeo telephoned the Saudi crown prince and also asked him to help arrange a ceasefire during Eidul Azha. The United States hopes that the ceasefire will enable the Taliban to experience the blessings of peace while celebrating the festival with their families.\nMr Pompeo, who will be arriving in Islamabad after the ceasefire, is expected to urge the new Pakistani government to back its efforts for bringing a durable peace in Afgha\xadnistan. In return for Pakistan’s support in Afghanistan, Washington may drop its opposition to a $12 billion aid package with the IMF and consider restoring its security assistance to Pakistan.\nAfghanistan was on Monday awaiting the Taliban’s response to President Ghani’s proposal for a three-month ceasefire, an offer welcomed by the US and Nato after nearly 17 years of war, according to AFP.\nThe president said his office had cleared “all obstacles” to peace with the announcement following consultations with religious scholars, political parties and civil society groups.\nThe Taliban did not immediately respond to President Ghani’s truce offer, but vowed to release “hundreds” of “enemy prisoners” to mark the Eidul Azha holiday. A Taliban member told AFP that the leadership had yet to issue a formal response to the ceasefire, but suggested fighting might be restrained during Eid even if no announcement was made."
            #     },
            # ]

            news = [
                {"text": "In Python, calculate how much time I spend on my phone per week!"},
                {"text": "In Python, calculate my coffee consumption"},
                {"text": "Push changes to a GitHub repository"},
            ]

            perplexity = load("perplexity", module_type="metric")

            for item in tqdm(news):
                input_text = item["text"]
                args.default_prompt = input_text
                _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(
                    input_text, args, model=model, device=device, tokenizer=tokenizer
                )
                without_watermark_detection_result = detect(decoded_output_without_watermark, args, device=device, tokenizer=tokenizer)
                with_watermark_detection_result = detect(decoded_output_with_watermark, args, device=device, tokenizer=tokenizer)

                item["wm_text"] = decoded_output_with_watermark
                item["len_wm_text"] = len(decoded_output_with_watermark)

                for sublist in without_watermark_detection_result[0]:
                    if sublist[0] == "z-score":
                        item["without_wm_scores"] = float(sublist[1])
                        break

                item["no_wm_text"] = decoded_output_without_watermark
                item["len_no_wm_text"] = len(decoded_output_without_watermark)

                for sublist in with_watermark_detection_result[0]:
                    if sublist[0] == "z-score":
                        item["with_wm_scores"] = float(sublist[1])
                        break

                try:
                    results = perplexity.compute(
                        model_id="facebook/opt-1.3b",
                        add_start_token=False,
                        predictions=[decoded_output_with_watermark, decoded_output_without_watermark],
                    )

                    item["ppl_score_wm"] = round(results["perplexities"][0], 2)
                    item["ppl_score_no_wm"] = round(results["perplexities"][1], 2)
                except:
                    continue

            with open("result_code.json", "w") as fp:
                json.dump(news, fp)

        if False:
            with open("result.json", "r") as fp:
                data = json.load(fp)

            roc_auc_scores = list()
            for i, r in [(1, 10), (1, 25), (3, 10), (3, 25)]:
                noisy_data = [
                    getNoisyData(item["no_wm_text"], item["wm_text"], i, r)
                    for item in data
                    if len(item["no_wm_text"]) > 0 and len(item["wm_text"]) > 0 and "with_wm_scores" in item
                ]
                noisy_data_results = [detect(item, args, device=device, tokenizer=tokenizer)[0] for item in noisy_data]

                z_score_cp = list()
                for item in noisy_data_results:
                    for subitem in item:
                        if subitem[0] == "z-score":
                            z_score_cp.append(float(subitem[1]))
                z_scores_baseline = [
                    p["with_wm_scores"] for p in data if len(p["no_wm_text"]) > 0 and len(p["wm_text"]) > 0 and "with_wm_scores" in p
                ]
                roc_auc = get_roc_auc_scores(z_score_cp, z_scores_baseline)
                roc_auc_1 = get_roc_auc_scores(z_score_cp, np.zeros_like(z_score_cp))

                len_noisy_data = [len(text) for text in noisy_data]

                roc_auc_scores.append(
                    {
                        "num_instances": float(i),
                        "ratio_wm_text": float(r),
                        "roc_auc": float(roc_auc),
                        "roc_auc_1": float(roc_auc_1),
                        "z_scores_cp": z_score_cp,
                        "avg_T": float(np.mean(len_noisy_data)),
                        "min_T": float(np.min(len_noisy_data)),
                        "max_T": float(np.max(len_noisy_data)),
                    }
                )

            with open("roc_auc_cp.json", "w") as fp:
                json.dump(roc_auc_scores, fp)

        if args.llm_attack:
            with open("result.json", "r") as fp:
                data = json.load(fp)
            llm_results = list()

            for item in data:
                if len(item["wm_text"]) > 0 and "with_wm_scores" in item:
                    paraphrased_text = list()
                    for prompt in prompts:
                        paraphrased_text.append(llm_paraphrasing(prompt, item["wm_text"]))

                    paraphrased_data_results = [detect(text, args, device=device, tokenizer=tokenizer)[0] for text in paraphrased_text]
                    z_score_cp = list()
                    for p_text in paraphrased_data_results:
                        for subitem in p_text:
                            if subitem[0] == "z-score":
                                z_score_cp.append(float(subitem[1]))

                    z_scores_baseline = [
                        p["with_wm_scores"] for p in data if len(p["no_wm_text"]) > 0 and len(p["wm_text"]) > 0 and "with_wm_scores" in p
                    ]
                    roc_auc = get_roc_auc_scores(z_score_cp, z_scores_baseline)

                    len_noisy_data = [len(text) for text in paraphrased_text]

                    llm_results.append(
                        {
                            "text": item["wm_text"],
                            "paraphrased_text": paraphrased_text,
                            "roc_auc_score": float(roc_auc),
                            "z_scores_cp": z_score_cp,
                            "avg_T": float(np.mean(len_noisy_data)),
                            "min_T": float(np.min(len_noisy_data)),
                            "max_T": float(np.max(len_noisy_data)),
                        }
                    )

            with open("roc_auc_llm.json", "w") as fp:
                json.dump(llm_results, fp)

        # Launch the app to generate and detect interactively (implements the hf space demo)
        if args.run_gradio:
            run_gradio(args, model=model, tokenizer=tokenizer, device=device)

        return


if __name__ == "__main__":

    args = parse_args()
    # print(args)

    main(args)
