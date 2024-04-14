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

genai.configure(api_key="AIzaSyAcL7v7OLOcBj4quiWdB_sMQxffVOXNltE")
model = genai.GenerativeModel("gemini-pro")

prompts = [
    "paraphrase the following paragraphs:\n",
    "“paraphrase the following paragraphs and try your best not to use the same bigrams from the original paragraphs:\n",
    "paraphrase the following paragraphs and try to keep the similar length to the original paragraphs:\n",
    "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n",
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases: \n",
]


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
        news = [
            {
                "text": "Review: This double disc is a complete overview of pianist Dharmawan's stunningly broad stylistic span. Not everything here is world-centric and it is not always pretty, but the gems are worth it, where he addresses the relationships between Indonesian music & modern jazz in unexpected and startlingly creative ways. Mark Wingfield adds some sonic guitar in places."
            },
            {
                "text": "A clinical-stage biotechnology company, engaged in the research and development of cancer therapeutics. Its mission is to research, develop and commercialize targeted cancer drugs with reduced toxicities compared to conventional cancer chemotherapeutics.\nHow do you think NASDAQ:ARQL will perform against the market?\nRead the most recent pitches from players about ARQL.\nI'm having a hard time reconciling a decent reason not to risk this stock with real money. In my opinion, at $2/share the company is poised to repeat miraculous gains in a relatively near future. Only twice in the past eleven years has this stock been this low. Its price has climbed up to as high as $10 in times past! With the company's current pipeline, a plausible economic crisis (something this sector can be relatively immune to and could therefore draw investors), and add in the fact that ArQule is presenting at the RBC Capital Markets Global Healthcare Conference coming this Tuesday... it seems likely that something positive, no matter how minute, will likely send this stock up quick. Dilution, or other bad things could happen, but I'm having a difficult time seeing it happen with this company. The fundamentals on this stock are shaky, though. This is not without risk.\nFind the members with the highest scoring picks in ARQL.\nThe Score Leader is the player with the highest score across all their picks in ARQL."
            },
            {
                "text": 'The location of the July 15, 2002 flare is shown at left. The other panels compare the scale of Earth to the eruptions. Red shows superheated gas held together by magnetic fields. The time sequence lasts only 80 seconds and yet reveals tremendous amounts of gas leaving the Sun.\nA detailed study of a huge solar eruption reveals that a series of smaller explosions combined in a domino effect to fuel the blast.\nThe findings improve understanding of the Sun\'s most powerful events and could lead to better forecasting of the tempests, researchers said.\nScientists studied data collected from a radiation flare on the Sun on July 15, 2002. The eruption, ranked as an X-7, was one of the most powerful in recent times. The flare was accompanied by a coronal mass ejection, which is a colossal discharge of electrified gas called plasma. The event was 5,000 million times more powerful than an atomic bomb.\nScientists don\'t know exactly what triggers such eruptions. They are associated with strong magnetic fields, however, and emanate from sunspots, which are cooler regions of the Sun that correspond to bottled-up magnetic energy.\n"Sunspots are at the surface of the Sun, and are essentially the footprints of the magnetic field," explained Louise Harra of the Mullard Space Science Laboratory at University College London. "The magnetic field reaches into the outer atmosphere in the same way as for example a bar magnet has a magnetic field around it."\nResearchers had thought the big eruptions are created when magnetic field lines from the core of a sunspot become tangled and reconnect high in solar atmosphere, or corona. The new study contradicts that assumption.\nX-7 flare started when plasma from below the Sun\'s surface broke suddenly through.\n"Below the surface of the Sun a dynamo process is working creating magnetic field," Harra explained in an email interview. "When this becomes buoyant it can rise to the surface of the Sun, and into the atmosphere."\nThe plasma collided with a strong magnetic field at the surface, and the interaction triggered release of "phenomenal amounts of energy," the researchers concluded. There were three eruptions, each triggering the next.\nThe gas was heated to 36 million degrees Fahrenheit (20 million Celsius) before being flung up into the solar atmosphere at 90,000 mph (40 kilometers per second).\n"We have observed the flows of hot gas for the first time, enabling us to see that several small flares combine to create a major explosion," Harra said. "This may eventually enable us to predict large flares before they erupt."\nNot all solar flares are accompanied by coronal mass ejections, and nobody knows for sure why.\n"It must be a combination of the magnetic field strength and the magnetic configuration that will allow field lines to be opened and hence the release of gas," Harra said.\nThe observations were made with SOHO spacecraft, a joint project of NASA and the European Space Agency. The results were presented last week at a meeting of the Royal Astronomical Society.'
            },
            {
                "text": "Happy Sunday. FRONT PAGE EDITORIAL -- BIRMINGHAM NEWS, HUNTSVILLE TIMES, PRESS-REGISTER (MOBILE) -- “STAND FOR DECENCY, REJECT ROY MOORE”: “This election is a turning point for women in Alabama. A chance to make their voices heard in a state that has silenced them for too long.\n“During the phone call on Wednesday afternoon, Mr. Ryan, who had campaigned heavily for Mr. Johnson in 2016, posed an essential question, according to the senator: ‘What are you going to need?’ What Mr. Johnson needs … is for the bill to treat more favorably small businesses and other so-called pass-through entities -- businesses whose profits are distributed to their owners and taxed at rates for individuals. Such entities, including Mr. Johnson’s family-run plastics manufacturing business, account for more than half of the nation’s business income, and the senator says the tax bill would give an unfair advantage to larger corporations.\n-- IT’S WORTH NOTING: This is hardly the first time Johnson has clashed with Senate Majority Leader Mitch McConnell and his GOP leadership team. He also fought with them over how the Obamacare repeal process played out. He is just the first Senate Republican out of the gate opposing the bill. Just because the House GOP tax overhaul was on the fast track and didn’t face many hiccups, don’t expect the same to be true in the Senate.\nFOR EXAMPLE… JAKE TAPPER talks with SEN. SUSAN COLLINS (R-MAINE) on CNN’S “STATE OF THE UNION: TAPPER: “You said this week that Republicans made a big mistake when they changed the tax bill to include this repeal of the Affordable Care Act individual mandate because that -- removing that could raise taxes or payment -- health care payments, premiums, for millions of Americans. If that provision stays in the tax bill, will that mean a ‘no’ vote from you?” COLLINS: “Well, first of all, I think we need to distinguish between taking away insurance from people who already have it, which is what the health care bill said we considered earlier this year would have done, versus removing a fine on people who choose not to have insurance. And that’s … disproportionately 80 percent on those who make under $50,000.\nMORE TAX DRAMA IN THE STATES -- “In Democrat-led state capitals, GOP tax reform push could scramble fiscal plans,” by Laura Nahmias in New York, Katherine Landergan in New Jersey and Carla Marinucci in California: “The Republican tax reform push in Washington is setting off budgetary alarm bells in high-tax states like New York, California and New Jersey, in the latest political skirmish to pit national Republicans against Democratic state and big city leaders.\n“With Republicans intent on shrinking or repealing the state and local tax deduction, California officials are worried that the House-passed tax bill, and the emerging Senate measure, will force local governments to reduce taxes and make big cuts to schools and social services. In New York, where New York City and state revenues are heavily reliant on just a handful of wealthy tax filers, budget watchdogs fear federal tax changes could trigger the flight of those residents. And in New Jersey, plans for a new millionaire’s tax, one of incoming Gov. Phil Murphy’s biggest campaign promises, are already being reined in as the Democratic-led New Jersey Senate waits on the outcome of any federal tax plan.\nTROUBLE FOR FRANKEN -- A1 of the STAR TRIBUNE -- “Sidelined by scandal, Sen. Al Franken faces questions about ability to do his job,” by Jennifer Brooks and Erin Golden: “Suddenly a senator whose statewide approval rating stood at 58 percent in the last Star Tribune Minnesota Poll is facing calls to resign — even from prominent Minnesota DFLers and deeply disappointed supporters.\n-- TOO CLOSE TO THE SUN?: STAR LEDGER FRONT PAGE: “Why this N.J. Republican keeps voting for things that could hurt Jersey” (print headline: “MacArthur is showing affinity for risk-taking: Representative’s tax stance could hurt N.J., his future”): “In just his third term in office, Rep. Tom MacArthur is helping to shape legislation in a way lawmakers who've been here a long time can only dream of.\nBUT, BUT, BUT -- EMILY HOLDEN in Bonn, Germany: “The White House goaded activists at the international climate talks by pushing coal and other fossil fuels. But behind closed doors, U.S. negotiators stuck to their Obama-era principles on the 2015 Paris deal — despite President Donald Trump’s disavowal of the pact. State Department negotiators at the U.N. conference that ended Saturday hewed to the United States’ long-established positions on the details of how to carry out the Paris agreement. And that's the U.S. role that most foreign political leaders sought to highlight, despite the low expectations inspired by Trump’s ‘America First’ agenda and his dismissal of human-caused climate change as a hoax.\n-- TONIGHT ON “KASIE D.C.”: Sen. Kirsten Gillibrand (D-N.Y.), former Alabama Gov. Don Siegelman, Katty Kay, Ashley Parker, Jonathan Swan, Leigh Ann Caldwell, Ken Dilanian, Paul Kane, Paul Singer, Gen. Barry McCaffrey and Azmat Khan.\nGREAT STORY -- SCOTT BROWN IN NEW ZEALAND -- “Scott Brown’s pay is $155,000 per year. The benefits are priceless,” by the Boston Globe’s Joshua Miller in Wellington, New Zealand: “Of the waves that followed from Donald Trump’s 2016 tsunami, Brown’s ascension from the everyman-with-a-pickup who lost two U.S. Senate races in two years in two states to US ambassador to New Zealand ranks among the most unlikely. And, for him, the most fortunate.\n“The island nation is a paradisiacal land of jade hills dotted with grazing sheep, golden-sand beaches surrounded by Jurassic Park-like jungles, snow-capped peaks that rise steeply from azure fjords, and pastoral villages serving gourmet meals and world-class wine.\n2020 WATCH -- “Don’t trust politicians to solve our problems, U.S. Sen. Ben Sasse tells Iowa crowd,” by the Des Moines Register’s Jason Noble: “Don’t look to politics to solve the pressing problems in American culture or address looming technological and economic changes that will rearrange American society, U.S. Sen. Ben Sasse told an Iowa audience Saturday. Politicians, he said, simply aren’t up to the task.\nSPOTTED: Sen. Susan Collins (R-Maine) at the Elton John concert in Bangor, Maine, last night. “She seemed to particularly enjoy his rendition of ‘Daniel,’” per our tipster.\n--SPOTTED: Wolf Blitzer and Dana Bash cutting the challah, Spencer Garrett, David Chalian, Alex Moe and Derek Flowers, Joy Lin, Juana Summers, John Legittino, Lauren Pratapas, Polson Kanneth and Sandhya Kotha, Ben Kochman, Rob Yoon, Katie Hinman, Sean and Ashley Kennedy.\nOUT AND ABOUT -- SPOTTED at the So Others Might Eat Gala Saturday night at the National Building Museum: Tom Donohue, Suzanne Clark, Bill Conway, Jack Gerard, Matthew Say, Jim McCarthy, Wayne Berman and the winners of this year’s Humanitarians of the Year award, Jane and Steve Caldeira of the Consumer Specialty Products Association.\n-- SPOTTED at the “Wonder Woman”-themed birthday party for BBC’s Suzanne Kianpour at Lapis Saturday night: Andrew Rafferty, Neil Grace, Molly Weaver, Walt Cronkite Jr., Lauren French, Paul Kane, Lauren Culbertson, Anastasia Dellaccio, Nikki Schwab, Brendan Kownacki, Sophie Pyle, Chris Brown, Lindsay Walters, Sean Weppner and Richard Strauss.\n-- Bert Gomez threw a party celebrating wife Susie Santana’s birthday Saturday night on the W hotel rooftop where guests salsa danced till midnight and were treated to cupcakes and the “Susie Q” specialty cocktail, according to a tipster. SPOTTED: Estuardo Rodriguez, Lyndon Boozer, Maria Cardona, Angela Arboleda, Laurie Saroff, Cristina Antelo and Miguel Franco.\n... Brad Bauman … Jason Dumont … John Axelrod, MSNBC alum now at BerlinRosen, is 26 ... Lauren McCulloch of “Meet the Press” ... Obama DOJ alum Dena Iverson DeBonis … Chris Harlow ... Eric Finkbeiner ... Mike Deutsch, FAA attorney … Matt Allen ... Beth Mickelberg … Lynne Walton ... Patrick K. O’Donnell ... Andrew Sollinger, EVP of subscriptions at Business Insider... Cait Graf, VP of comms at The Nation ... Ivan Levingston ... Alexander Heffner is 28 ... Ellen Silva of NPR ... Shelley Hearne (h/ts Jon Haber) … Charlie Siguler ... Geoff Sokolowski ... Neil Bjorkman, VP of legislative affairs at the U.S.-India Strategic Partnership Forum … Hannah McLeod … Michael Reynold … Amber Manko … Bush 43 W.H. alum Ivvete Diaz ... Bush 43 HHS alum Mary Kay Mantho, now director at GSK ... Ivette Diaz ... Shannon Vilmain ... Barb Leach ... Julie Cassidy … David O’Boyle ... Ricky Wilson."
            },
            {
                "text": "Under intense pressure to improve conditions in the jail complex on Rikers Island, the administration of Mayor Bill de Blasio has developed a plan to move 16- and 17-year-olds to a dedicated jail for youths in the Bronx.\nThe cost to carry out the plan is expected to be about $300 million, officials said.\nThe plan calls for the city to reconfigure the Horizon Juvenile Center, which is currently used to hold 14- and 15-year olds, to house the 16- and 17-year olds who are typically sent to Rikers.\nA 2015 settlement with the Department of Justice on reform at Rikers called on the city to seek an alternative location to house inmates under 18, although it stopped short of requiring it.\nNew York is the only state other than North Carolina that prosecutes all children 16 and older as adults if they are accused of a crime. In New York City, they are likely to wind up at Rikers, a notoriously brutal lockup. There are currently about 200 inmates who are 16 or 17 at Rikers, down from about 330 in 2013.\nThe city and many advocates have urged state lawmakers to pass a law that would treat those under the age of 18 as juveniles, but the proposal has gone nowhere in Albany.\nThe new plan to move 16- and 17-year-olds from Rikers must overcome several hurdles. It has to be approved through the city’s time-consuming land use process: The local community board and the Bronx borough president get to weigh in and it must be approved by the City Planning Commission and the City Council.\nBut the change will not happen any time soon; officials said it could take four years or more to get approvals and to complete construction. The plan also calls for the city to remodel another juvenile detention site, the Crossroads Juvenile Center in Bushwick, Brooklyn, to hold all of the city’s 14- and 15-year-old detainees, including those who had previously gone to Horizon.\nThe cost of refurbishing the Bronx center is budgeted at $170 million. The cost of remodeling the Brooklyn center is budgeted at $129 million.\nAdvocates supported the move but lamented the long delay before the new center will be ready, assuming that it survives the land-use review process.\n“For us this is a marathon, not a sprint.” he said.\nMore than 95 percent of the 16- and 17-year-olds at Rikers are awaiting trial. More than a third have been charged with robbery and about one in 10 have been charged with assault, according to data provided by the city.\nIncreased attention was focused on the plight of younger teenagers at Rikers in 2014 after The New Yorker published an article about Kalief Browder, who was sent there at 16, accused of stealing a backpack. He never stood trial or was found guilty of any crime but he spent three years at Rikers, nearly two of them in solitary confinement. He told of being beaten repeatedly by guards and other inmates and trying several times to kill himself while in custody. After his release he remained deeply troubled by the experience and he committed suicide last year at age 22.\nThe city ended solitary confinement for Rikers inmates under 18 in December 2014."
            },
            {
                "text": "It’s dense, interesting, weird and funny, but what it’s all wrapped around is another “Chosen One” story that really feels drawn out and convoluted, crumbling into something tiresome and tedious.\nWhere the science-fiction genre is concerned, it’s not hard to at the very least appreciate the ambition of the storytelling. It takes an honest, extra layer of creativity to create a reality not bound by the restrictions of our own, and it can be a delight to watch a perfect mixture of detailed universe building mesh with smart, unique storytelling. Of course, this requires delicate balance, as a dip too far one way finds a convoluted mess, and a dip in the opposite direction creates something that ultimately feels hollow and incomplete. It is sadly the former trap to which Andy and Lana Wachowski’s Jupiter Ascending falls victim.\nIn a similar vein to what they were able to accomplish with previous sci-fi stories like The Matrix and Cloud Atlas, Jupiter Ascending is a narrative set in a vivid and comprehensive world. There are legions of soldiers who have had their DNA spliced with animals – like the wolf-infused Caine (Channing Tatum) or the bee-like Stinger (Sean Bean). There are galaxy-spanning businesses where worlds are held like long-term investments. There’s even intergalactic low-level bureaucracy that has all varieties of individuals waiting in lines and shuffling from one department to the next. It’s dense, interesting, weird and at times funny, but what it’s all wrapped around is another “Chosen One” story that really feels drawn out and convoluted, eventually crumbling into something tiresome and tedious – albeit consistently beautiful.\nThe Wachowskis do legitimately make good choices in the casting department, as both Mila Kunis and Channing Tatum make for fun, charismatic leads to take us through the story. Kunis stars as Jupiter Jones, an illegal alien living with her Russian immigrant family and working as a housekeeper living in modern day Chicago. As we see happen oh so often at the cinema, however, her life changes in an instant when she meets Caine (Tatum), an intergalactic bounty hunter. Jupiter is not only made aware of life on other planets, but also that she is the genetic recurrence of one of the most powerful women in the galaxy, and stands to literally inherent the Earth now that she has passed away. It’s a simple and rather familiar plot, but through a first act of enjoyable “awakening” material and sky-shredding action scenes, Jupiter Ascending holds its own.\nIt’s with the introduction of the trio of central villains that everything begins to fall apart – which is a sincere problem considering that they wind up completely driving the entire story. These are the three surviving members of the Abrasax family -- Balem (Eddie Redmayne), Titus (Douglas Booth) and Kalique (Tuppence Middleton) – and the sons and daughter of the woman whom Jupiter Jones is a genetic equal. Each of them has a personal desire to take control of the Earth – which is alternately described as being extremely valuable and just a single piece of a much larger business – but also each take their own place doing more harm to the movie than good. Kalique winds up being mostly useless to the story, simply providing huge amounts of exposition and then disappearing; Titus merely feels like a stall in the narrative, serving as what is really a fairly pointless obstacle; and Balem simply feels overcooked in every sense, with Redmayne giving a highly bizarre, scenery-chewing performance full of needless affectation that makes him stick out like a sore thumb (really making you wonder what kind of off-base directions he was being given by the Wachowskis). As Jupiter Ascending’s plot encounters these characters, the film begins to degrade faster and faster, exploring needless and unnecessary elements of the sci-fi world that do more subtraction than addition.\nAt this point in their careers, Andy and Lana Wachowski have become better known for their visual flash and style, and it serves the movie well. Their flair is apparent from the costume and makeup work (the idea of DNA splicing works magic in this department) to the production design, and it all serves to provide Jupiter Ascending with a unique feel (even when it wears references to movies like Brazil on its sleeve). Immense credit also goes to the visual effects teams, who not only render some fantastic air battles and spaceships, but also expansive alien environments that do their part to make the audience feel like they’re being taken to a different world. All of this spectacle doesn’t make up for the fact that character motivations are at times completely dubious, and that the third act drags on far too long, but it does help.\nIt’s hard to really begrudge a film like Jupiter Ascending, both for its creative scope and in that it is one of the few-and-far-between female-led action movies, but it is equally hard to validate it beyond those elements, it’s aesthetics, and the casting of the leads. There is perhaps a better feature that exists partially on the cutting room floor – explaining some of the more bizarre plot developments/holes – or one that could have been made with just a few tweaks to the screenplay. But the movie that will be hitting theaters this week is best described as unfortunate."
            },
            {
                "text": "Advocates for government accountability were alarmed to learn Gov. Jared Polis was undecided on whether to veto a bill to expand police transparency in Colorado.\nBut Polis spokeswoman Maria De Cambra set the record straight late Wednesday.\nThe signing is slated for 10:50 a.m. Friday at the Capitol, and it will be open to media, De Cambra said.\nPolis is allowed to wait 10 days to sign bills, which gives him until Saturday in this case. In an earlier call Wednesday, De Cambra had said that the governor was still undecided.\nThe bill — HB-1119, sponsored by Democratic Denver Rep. James Coleman — would require Colorado law enforcement agencies to open the files on completed internal investigations into a wide range of police interactions with citizens, including alleged incidents of excessive force.\nThis would mean the public, media, lawyers and any other interested parties would, through open records requests, get a glimpse at how Colorado cops police themselves. As it stands, the Denver Police Department is the only Colorado law enforcement agency that consistently releases comprehensive information following internal investigations.\nThe bill passed the House and Senate and has been sitting on the governor’s desk for about a week, awaiting the final signature needed to make HB-1119 state law.\nUpon learning Polis was not necessarily on board with the bill, the Colorado Press Association — one of the primary backers of the bill — sent out an email to its members that called the situation “urgent” and implored journalists and editorial boards to speak out now in support of HB-1119.\nMedia in the state have heralded the bill as a key step toward transparency from an institution that shrouds much of its accountability system in secrecy. The Denver Post’s Noelle Phillips testified in support of HB-1119 this year.\nThe effort’s other supporters include the ACLU and Colorado Common Cause, the libertarian Independence Institute and media groups such as the Colorado Broadcasters Association and the Colorado Freedom of Information Coalition. Colorado Independent Editor Susan Greene testified last year in support of a version of the bill."
            },
            {
                "text": "Two recently published reports shed damning light on the high cost of low wages in the fast food industry – an industry dominated by the restaurant giant McDonald’s, which raked-in over $5.4 billion in profits last year.\nThe reports – Super Sizing Public Costs: How Low Wages at Top Fast-Food Chains Leave Taxpayers Footing the Bill and Fast Food, Poverty Wages: The Public Cost of Low-Wage Jobs in the Fast-Food Industry – argue that low-wages in the fast-food industry cost taxpayers nearly $7 billion annually.\nMedicaid and CHIP (the Children’s Health Insurance Program) account for more than half of the $7 billion, at an average of $3.9 billion annually. Additionally, due to low earnings, fast food workers’ families also receive an annual average of $1.04 billion in food stamp benefits and $1.91 billion in Earned Income Tax Credit payments.\nThe ten largest fast-food companies alone made more than $7.4 billion in profits in 2012 and paid their top executives more than $53 million in compensation.\nLow-wage fast-food jobs cost tax-payers the most in California ($717 million), New York ($708 million), Texas ($556 million), Illinois ($368 million), and Florida ($348 million).\nAccording to Super Sizing Public Costs, “McDonald’s represents the most costly fast-food company for tax-payers.” Low wages and lack of benefits at McDonald’s cost tax-payers “$1.2 billion every year in public assistance programs,” according to the report.\nYUM! Brands (Pizza Hut, Taco Bell, KFC), Subway, Burger King and Wendy’s round-out the top five fast-food companies with workers who rely on public assistance.\nThe median average wage for fast-food workers is $8.69; an estimated 87 percent of fast-food workers do not receive health benefits. Furthermore, 67 percent of front-line fast-food workers are adults 20 and older; 68 percent are the main earners in their families and more than one-quarter are raising children.\n“This business model puts tax-payers on the hook, while rewarding corporate CEO’s,” Jack Temple, a policy analyst for the National Employment Law Project and co-author of Super Sizing Public Costs, said during a conference call with the press on October 15.\nThe reports come on the heels of a nation-wide wave of fast-food worker strikes; most recently strikes took place in over 60 cities across the nation with thousands of fast-food workers walking off their jobs.\n“The CEO of McDonald’s makes more in a day than I make in a year,” Yates added.\nFast-food workers nationally are calling for $15 an-hour in pay and the right to form or join a union without retaliation.\nJacobs also addressed the wave of non-traditional worker organizing taking place in fast-food, retail and other low-wage service sector jobs.\nSuper Sizing Public Costs: How Low Wages at Top Fast-Food Chains Leave Taxpayers Footing the Bill was published by the National Employment Law Project and Fast Food, Poverty Wages: The Public Cost of Low-Wage Jobs in the Fast-Food Industry was published by the University of California Labor Center."
            },
            {
                "text": "Attendees visit the Vodafone Group pavilion at the Mobile World Congress in Barcelona, Spain, March 4, 2015.\nVodafone, one of the world's biggest telecoms companies, announced better-than-expected sales Friday as its 4G plans grow.\nThe U.K.-based company reported an acceleration in its main quarterly sales growth on Friday as a cable TV acquisition in its biggest market, Germany, where it faces strong competition from Deutsche Telecom, lifted sales and consolidated the overall return to growth for the British firm. It also returned to growth in the U.K.\nHowever, there was no mention in its statement of the one thing many investors want to hear about - a potential deal in Western Europe with John Malone's Liberty Global. The U.K., Germany and Netherlands were the markets identified by Malone as most fertile ground for a tie-up in May, but little has been heard on the deal in recent weeks.\nThe world's second-largest mobile operator said first-quarter organic service revenue grew 0.8 percent, ahead of the 0.1 percent it recorded in the fourth quarter and better than most analysts had expected.\nVittorio Colao, chief executive of the company, said in a statement: \"Our emerging markets have maintained their strong momentum and more of our European businesses are returning to growth, as customer demand for 4G and data takes off,\" said.\nThe group reiterated its outlook for the full year."
            },
            {
                "text": 'The Volstead Act was repealed on this date in 1933, returning control of alcohol to the states, and it was a happy day at the F.X. Matt Brewery.\nUTICA, N.Y. -- A total of 85 years ago Wednesday marked the end of prohibition, which meant legally drinking beer again. And for the Matt family from Utica – that meant getting back to selling beer again.\nThey let customers and suppliers know that at midnight, they could deliver, buy, and drink beer at the brewery once again.\nThe local business managed to survive 13 years.\n"Kept it going with soft drinks and malt tonic and malt syrups, but this was a special day. In fact, he sent a letter out to announce beer was available at midnight, the next day, actually,” said F.X. Matt Brewery CEO/Chairman Nick Matt.\nMatt says sales dropped off 75 percent when prohibition went into effect. And thanks to the current craft beer movement, business has been growing by leaps and bounds.'
            },
            {
                "text": "NORTH BRUNSWICK, N.J. — Body cameras captured the moment a North Brunswick police officer climbed up a burning building to rescue a family.\nThe fire started Saturday around 10:20 P.M. at the Governor's Pointe Condominiums. When Officer Grasso arrived, two grandparents and their grandchildren were trapped on the balcony of their apartment.\nGrasso climbed up to the family and handed the family members down to other officers who were on the ground below.\nIn all, 28 people were left homeless after the fire destroyed 12 apartments. No one was injured."
            },
            {
                "text": 'Check out the Charleston Parks Conservancy\'s newest glow-up project at their inaugural Hampton Park culinary event, Shucked + Sauced, Sat. Jan. 26 from 1 to 4 p.m.\nThe event will take place at the newly renovated Rose Pavilion, a previously unused concession stand that has been "re-envisioned and refurbished through the collaborative efforts of the Charleston Parks Conservancy and the City of Charleston." This space will serve as a gathering spot for community events and park programs.\nConventional wisdom says that oysters are best enjoyed during those months containing the letter "r," making September through April prime oyster season.\nShucked + Sauced was curated by chef Mike Lata of FIG and The Ordinary and will feature bites from The Ordinary, Nico, 167 Raw, Chubby Fish, Rodney Scott\'s BBQ, Lewis Barbecue, and The Obstinate Daughter in addition to local oysters from Lowcountry Oyster Co., Clammer Dave, Barrier Island Oyster Co., and Lady\'s Island Oysters. There will be wine, and beer from Holy City Brewing Co., plus beachy Lowcountry tunes from Dallas Baker and Friends.\nLata said in a press release, "This event connects two of my favorite things: oysters and Charleston. Having been here for 20 years, I\'ve seen many changes in Charleston. As the city has grown, development has put a lot of stress on the fishing community. Our local oyster farmers are helping to revitalize our working waterfront, just as the Parks Conservancy works to rejuvenate and preserve our city\'s public green spaces."\nTickets are $85 and include all food and beverages. All proceeds from this event support the Conservancy and their work renovating and beautifying public parks and green spaces.\nMore photos from our visit to Lowcountry Oyster Co.'
            },
            {
                "text": "OKLAHOMA CITY - Kevin Durant scored 25 points, James Harden added a spark with 23 points off the bench while returning from an injury and the Oklahoma City Thunder ended a string of futility against the Houston Rockets with a 122-104 victory.\nThe Thunder got out to a sizzling start and opened a 23-point lead within the first 14 minutes. Their lead was never in jeopardy against a Houston team desperately needing a late charge to make the playoffs.\nInstead, Oklahoma City used its second-best shooting performance of the season to take a big step toward solidifying its spot in the postseason.\nLuis Scola scored 25 and Trevor Ariza added 20 points for Houston, which fell seven games behind Oklahoma City and 51/2 behind eighth-place Portland in the Western Conference.\nBOSTON - Paul Pierce scored 27 points, Kevin Garnett had 20 and the Boston Celtics clinched a playoff berth with a win over the Denver Nuggets.\nThe Celtics dominated inside and the Nuggets, one day after losing to the Knicks in New York, slipped a half-game behind the Dallas Mavericks and into a third-place tie with Utah in the Western Conference with their third consecutive loss.\nBoston led by 21 points twice in the third quarter before Denver cut the lead to 87-80 going into the fourth. But the Celtics stayed ahead by seven to 16 the rest of the way for their fifth win in six games. Rajon Rondo had a triple-double with 11 points, 15 assists and 11 rebounds.\nCarmelo Anthony led Denver with 32 points and J.R. Smith added 21.\nTORONTO - Deron Williams had 18 points, 16 assists and eight rebounds, leading the Utah Jazz to victory over the Toronto Raptors.\nCarlos Boozer added 18 points and 11 rebounds, while Mehmet Okur and rookie Wesley Matthews had 16 points apiece for the Jazz, who won their third straight. Seven Jazz players finished with at least 12 points.\nChris Bosh celebrated his 26th birthday with a 20-point performance for the Raptors (35-35), who fell 11/2 games behind Charlotte and Miami in the race for sixth place in the Eastern Conference.\nAntoine Wright had 10 of his 15 points in the second quarter for the Raptors, while Andrea Bargnani added 12 and Jose Calderon 10.\nATLANTA - Josh Smith swooped in to slam through Joe Johnson's missed shot just ahead of the buzzer and the Atlanta Hawks finally beat the Orlando Magic, clinching a third straight trip to the playoffs.\nThe Hawks bounced back from an early 15-2 deficit and overcame going 8:45 in the fourth quarter without a field goal against a division rival that had routed them three times this season. For Atlanta, it came down to the final shot of regulation for the fifth straight game.\nINDIANAPOLIS - Danny Granger scored 31 points to help the Indiana Pacers beat the Washington Wizards and extend Washington's losing streak to a franchise record-tying 13 games.\nAndray Blatche started for the Wizards a night after he was benched for most of the game following a spat with coach Flip Saunders.\nThe Wizards last lost 13 straight in 1995. They can set a record for futility Friday at Charlotte.\nIt was Granger's third straight game with at least 30 points. Josh McRoberts had 14 points and a career-high 12 rebounds for the Pacers, who have won four in a row overall and six straight at home.\nCHARLOTTE, N.C. - Stephen Jackson shook off a sore hand and busted out of his shooting slump with 37 points, leading the Charlotte Bobcats past Minnesota to send the reeling Timberwolves to their 14th straight loss.\nJackson, who a night earlier revealed he had been plagued by a bruised ligament near his left index finger, hit 15-of-24 shots after shooting 37 percent in the previous 10 games. Gerald Wallace added 23 points as the Bobcats shot 54 percent from the field and moved into sixth place in the Eastern Conference with their sixth straight home win.\nAl Jefferson scored 21 points for the Timberwolves, who came apart in a 20-0 Charlotte run in the second quarter to drop to 5-32 on the road.\nEAST RUTHERFORD, N.J. - Brook Lopez had 26 points and 13 rebounds, and New Jersey Nets moved a step closer to avoiding a dubious date with NBA history by beating the Sacramento Kings to snap an eight-game losing streak and a franchise-record, 14-game skid at home.\nDevin Harris added 24 points and nine assists for the Nets (8-63), who need to win once in their last 11 games to avoid breaking the league mark for fewest wins in a season (9-73), set by the Philadelphia 76ers in 1972-73.\nNets interim coach Kiki Vandeweghe guided the team despite learning of the death of his mother, former Miss America Colleen Kay Hutchins, earlier in the day.\nBeno Udrih had 19 points for the Kings, who lost the eighth time in 11 games.\nMILWAUKEE - Rookie Jrue Holiday had 15 points and seven assists, leading the Philadelphia 76ers to a victory over the Bucks that snapped Milwaukee's longest home winning streak in six years at eight games.\nIt was a dismal display for Milwaukee, which had won 15 of the last 17 to virtually assure a postseason berth for the first time since 2006.\nInstead, Samuel Dalembert had 12 points and 10 rebounds, Willie Green scored 16 points and Andre Iguodala 14 as the Sixers poured it on for their third win in the last 16 games.\nJerry Stackhouse scored 15 points and rookie Brandon Jennings added 12 for Milwaukee, but the Bucks shot 5 of 28 from 3-point range and led only once in the game, 3-0.\nNEW ORLEANS - LeBron James scored an efficient 38 points on 15 of 22 shooting to go with nine assists, and the Cleveland Cavaliers won their eighth straight game, over the New Orleans Hornets.\nJ.J. Hickson scored 20 for the Cavs, who led by as many as 17 and never trailed after James' floater tied it at 10 in the first quarter. Delonte West added 15 points and Antawn Jamison had 11 points and 11 rebounds as Cleveland won its 27th straight over a team with a losing record.\nMarcus Thornton scored 20 points and Darren Collison added 17 for the Hornets, who will be eliminated from the playoffs if Portland wins today. David West added 16 points."
            },
            {
                "text": "Famously tight-lipped about her Scientology faith, The Handmaid’s Tale star Elisabeth Moss has opened up in a candid new interview.\nElisabeth Moss won’t delve into the specifics of her beliefs in Scientology, but she does think they perfectly align with her work on The Handmaid’s Tale, despite what critics may think.\n“Listen, it’s a complicated thing because the things that I believe in, I can only speak to my personal experience and my personal beliefs. One of the things I believe in is freedom of speech,” she told The Daily Beast.\n“I believe we as humans should be able to critique things. I believe in freedom of the press. I believe in people being able to speak their own opinions.\n“I don’t ever want to take that away from anybody, because that actually is very important to me,” she continued. “At the same time, I should hope that people educate themselves for themselves and form their own opinion, as I have.\n“The things that I believe in personally, for me, The Handmaid’s Tale, and the ability to do something that is artistically fulfilling but is also personally fulfilling, I’ve never had that.\nMoss, 36, said taking away the right to talk about certain topics or believing in certain religions would cause society to turn into a place that mirrors Gilead, the fictional dystopia that serves as the setting for The Handmaid’s Tale.\n“I don’t choose to talk about not just religion, but my personal life — who I’m dating and that kind of thing,” she said.\n“ … I think people should be allowed to talk about what they want to talk about and believe what they want to believe and you can’t take that away — and when you start to take that away, when you start to say ‘you can’t think that,’ ‘you can’t believe that,’ ‘you can’t say that,’ then you get into trouble. Then you get into Gilead.\nWhile the former West Wing star is passionate about her religion, there are also tenets of it in which she doesn’t believe, such as its history of having anti-LGBTQ views.\n“Which is not where I stand. It’s like, it’s a lot to get into and unpack that I can’t do. But that is not my bag,” she said."
            },
            {
                "text": "By the time Ohio State’s slow-arriving student section filled up, the Buckeyes had already taken a 7-0 lead against UNLV as speedy H-back Parris Campbell raced 69 yards for the opening touchdown on the offense’s second play. The Scarlet and Gray extended the early lead, continued to build upon it and never allowed the Rebels to even feign a threat as Ohio State dominated, winning 54-21 Saturday afternoon at Ohio Stadium.\nRedshirt senior quarterback J.T. Barrett marched his team down the field at will against an overmatched, less-talented UNLV defense, completing 12-of-17 passes for 209 yards and five touchdowns and subbed out before halftime.\nSeven players — wideouts Terry McLaurin, Johnnie Dixon, K.J. Hill, Binjimen Victor, Campbell and walk-on C.J. Saunders and tight end Rashod Berry — caught touchdowns for the Buckeyes, the most in a single game in Ohio State history.\nBarrett overthrew sophomore wideout K.J Hill on one of his first passes of the game, but settled in as the Buckeyes scored on all but one of his drives. Campbell led Ohio State with three catches for 105 yards, but fumbled near the goal line on his team’s third drive of the game.\nThe Rebels offense stood no chance facing off against the Buckeyes’ stout defense. An aggressive, blitz-heavy defensive front pressured redshirt freshman quarterback Armani Rogers the entire game. Late in the first quarter, backed up at the 2-yard line, defensive tackle Dre’Mont Jones stuffed a run and forced Ohio State’s first safety of the season.\nThe Buckeyes racked up four sacks and a season-high 13 tackles for loss. Sophomore defensive end Nick Bosa led the Buckeyes with three tackles behind the line of scrimmage.\nRogers competed 11-of-19 passes for 88 yards. The Rebels, buoyed by junior running back Lexington Thomas’ 55-yard touchdown, rushed for 41 yards on 176 carries.\nWith 3:32 left in the second quarter while leading 37-7, redshirt freshman quarterback Dwayne Haskins replaced Barrett, and first-team All-American center Billy Price subbed out of the blowout.\nHaskins threaded the needle to Saunders for his first touchdown of the game, a 28-yard strike across the middle. The strong-armed quarterback went 15-for-23 and 228 yards and tossed two touchdowns. He hit Berry late in the third quarter who rumbled for a 38-yard touchdown, the first of the defensive end-turned-tight end’s career.\nHaskins later threw an interception to linebacker/defensive back Javin White, who took it 65 yards for a touchdown, the first pick-six thrown by an Ohio State quarterback this year.\nFreshman running back J.K. Dobbins took 14 carries 95 yards. Once again, redshirt sophomore running back Mike Weber did not play. He has dealt with a hamstring injury since the beginning of fall camp and missed the first game of the season.\nDefensive tackle Robert Landers, offensive guard Matt Burrell, linebacker Chris Worley and cornerback Shaun Wade also did not play for Ohio State due to injuries.\nOhio State will look for its third consecutive victory when the Buckeyes head to Piscataway, New Jersey, next Saturday to take on the Rutgers Scarlet Knights (1-2) at 7:30 p.m.\nRewarded with another drop in the polls. Nothing will change until they beat a good team. Penn State will be a challenge. They had a great comeback against Iowa."
            },
            {
                "text": "SOUND FORGE Audio Cleaning Lab can be used to record any sound you can hear on your computer - from internet radio to LPs and tapes which have been connected to the sound card. When recording older material, particularly when it has been stored on vinyl or cassette, it is likely that the sound quality has deteriorated over time. Using this program you can remove any unwanted extraneous noise such as clicks, pops and hisses and generally improve the overall audio experience working with an easy to use interface.\nOf course, you don't need to record audio either. You can simply import an existing track of segment of music for processing, before exporting the final track.\nAudio Cleaning Lab will remove unwanted sections from a recording, increase the volume of quiet tracks and split longer recording up into multiple sections - in fact, Audio Cleaning Lab can take care of much of this for you automatically. A great level of control is available through the use of a graphic equaliser that can be used to balance out tone, while additional options are available to improve the quality of video sound, reduce the volume of background music on vocal tracks and much more besides.\nDon't worry, most of the options are presets which you can select to improve or clean your audio. There are options to reduce tape noise from old cassettes, reduce crackling from your LPs, remove wind noise from recorded voice, and even create your own presets to use later. You can increase the loudness of the track, increase the stereo width, add various special effects and much more.\nIf you've been looking for a way to digitise your music collection, Audio Cleaning Labs Deluxe may just be the tool you have been looking for. In a few simple steps you can record music to your computer, enhance its sound and then export the results as high-quality MP3s/OGG/FLAC/AAC and other formats that can be played back on a variety of devices.\nProfessional quality audio clean-up tool which will produce superb results from an instantly accessible program."
            },
            {
                "text": "Two people were killed and two others injured Monday in what authorities are calling a possible murder-suicide at an elementary school in San Bernardino, Calif.\nAccording to USA Today, Police Chief Jarrod Burguan said that two adults were dead and that two students had been wounded and taken to the hospital.\nDistrict spokesperson Maria Garcia described the shooting as a domestic dispute, according to the news site, and confirmed that at least two students had been airlifted to a hospital."
            },
            {
                "text": "Posted on Wed, Mar 14, 2012 : 8:40 a.m.\nA Manchester woman’s selfless gift could help extend the life of a co-worker’s husband, WXYZ reports.\nLisa Moutinho, an administrator at the Washtenaw County District Court in Ypsilanti, is giving one of her kidneys to Michael Brunson, who’s been on dialysis for five years, the station reported.\nThe surgeries are scheduled for Wednesday at Henry Ford Hospital in Detroit.\nThu, Mar 15, 2012 : 3:28 p.m.\nMy daughter has lived 37 years with one kidney. No problems all these years. Not even a kidney infecion. Congratulations to Lisa for her generosity and good future health to Michael!!!\nWed, Mar 14, 2012 : 6:03 p.m.\nI've known Lisa for many years and this doesn't surprise me in the least. Yeah Lisa!\nWed, Mar 14, 2012 : 3:15 p.m.\nThis is certainly an amazing gift. Usually it is a relative who does this. But, if something happens to her one-good kidney, then what? She also has 3 children.\nThu, Mar 15, 2012 : 1:41 a.m.\nThe fear you express in this question is one many people have. Part of the reason that I hear people asking this question is that folks are really unaware of basic biology. It's really uncommon for something to happen to &quot;one&quot; kidney. The most likely thing to have happen to one kidney, is to be born without it. Donating doesn't have a mathematically significant impact upon the donor.\nWed, Mar 14, 2012 : 5:20 p.m.\nHarry, I am an organ donor but maybe some people have religious objections? I just hope that people will consider doing it.\nWed, Mar 14, 2012 : 3:51 p.m.\nHopefully there are people who are as generous as she is. One thing I will never understand. Why isn't every person in america an organ donor. Your dead why would you want to be buried with your organs?\nWed, Mar 14, 2012 : 2:29 p.m.\nNow here is a hero. Its not her job. She's not getting paid. We seem to throw around that word a lot these days.\nWed, Mar 14, 2012 : 2:27 p.m."
            },
            {
                "text": "MADRID: As Cristiano Ronaldo enjoyed his time off at the French Riviera, his Portugal team-mates were busy handing Italy another disappointing result in the UEFA Nations League.\nPortugal did not need Ronaldo, who skipped international duty to recharge after a busy summer, to beat Italy 1-0 in Europe’s newest soccer competition.\nElsewhere, Turkey mounted a remarkable comeback from two goals down against Sweden, while Kosovo made history with their first competitive win since being admitted to UEFA in 2016.\nThe European champions easily outplayed Italy, which had opened its Nations League campaign with a disappointing 1-1 home draw against Poland in its first competitive match under coach Roberto Mancini.\nAndre Silva scored on a counterattack early in the second half to give Portugal the Group 3 win in the top-tier League A. The hosts had several chances to extend their lead at the Stadium of Light in Lisbon, while Italy created little and relied mostly on set pieces.\nThe Nations League gives UEFA’s 55 member countries competitive games and eliminates friendlies. The winners of the League A groups featuring the highest-ranked countries go into a final-four competition in June.\nSantos did not call up Ronaldo after talking to the player and saying he needed more rest following the World Cup and his transfer from Real Madrid to Juventus. Ronaldo also didn’t play in the team’s 1-1 draw in a friendly against Croatia last week.\nSocial media photos showed Ronaldo and his family enjoying time off in Monaco in recent days.\nIt was Italy’s second competitive match since last year’s World Cup playoff loss to Sweden.\nTwo goals five minutes apart early in the second half gave Kosovo a historic 2-0 win over the Faroe Islands in Group 3 of the fourth-tier League D.\nKosovo, the Balkan republic which was accepted by UEFA and FIFA two years ago, had opened with a 0-0 draw at Azerbaijan.\nTurkey bounced back from an opening loss to Russia by coming from two goals down to defeat Sweden 3-2 thanks to two late goals by Emre Akbaba.\nAkbaba scored in the 88th minute and two minutes into stoppage time to give Turkey the League B victory.\nSweden, making its Nations League debut following its surprising quarterfinal appearance at the World Cup, had taken a 2-0 lead by the 49th minute at Friends Arena in Stockholm.\nIt was Turkey’s first away win in seven matches.\nIn the third-tier League C, Scotland ended its run of three straight losses with a 2-0 win over Albania in Group 1.\nIn Group 4, Serbia and Romania drew 2-2, while Montenegro defeated Lithuania 2-0.\nIn League D, Andorra and Kazakhstan drew 1-1 in Group 1, while Malta drew with Azerbaijan 1-1 in Group 3.\nThe Nations League guarantees at least one of Europe’s low-ranked League D teams will qualify for Euro 2020 through a playoffs."
            },
            {
                "text": "ST. JOHNS COUNTY, Fla. - Commissioners in St. Johns County voted unanimously to postpone any legislative action against a localized ban of a substance that the DEA has classified as a \"drug of concern.\"\nAccording to a report by the St. Augustine Record, 19 speakers during a Tuesday night commission meeting defended their right to use kratom, speaking about the health benefits they'd recieved while battling fibromyalgia and anxiety. Commissioners agreed to table any action until at least May, after the Legislative session.\nORIGINAL STORY: Could drug with opioid-like properties be banned in St. Johns County?\nThe FDA has warned people against using kratom, saying it has properties similar to those of opioids. It's legal in the U.S. aside from Alabama, Arkansas, Indiana, Tennessee, Vermont, Wisconsin and the District of Columbia.\nThere are no restrictions on kratom in Florida, expect in Sarasota County, where it's banned."
            },
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

            results = perplexity.compute(
                model_id=args.model_name_or_path,
                add_start_token=False,
                predictions=[decoded_output_with_watermark, decoded_output_without_watermark],
            )

            item["ppl_score_wm"] = round(results["perplexities"][0], 2)
            item["ppl_score_no_wm"] = round(results["perplexities"][1], 2)

        with open("result.json", "w") as fp:
            json.dump(news, fp)

        if args.cp_attack:
            roc_auc_scores = list()
            for i, r in [(1, 10), (1, 25), (3, 10), (3, 25)]:
                noisy_data = [getNoisyData(item["no_wm_text"], item["wm_text"], i, r) for item in news]
                noisy_data_results = [detect(item, args, device=device, tokenizer=tokenizer)[0] for item in noisy_data]

                z_score_cp = list()
                for item in noisy_data_results:
                    for subitem in item:
                        if subitem[0] == "z-score":
                            z_score_cp.append(float(subitem[1]))
                z_scores_baseline = [item["with_wm_scores"] for item in news]
                roc_auc = get_roc_auc_scores(z_score_cp, z_scores_baseline)

                len_noisy_data = [len(text) for text in noisy_data]

                roc_auc_scores.append(
                    {
                        "num_instances": float(i),
                        "ratio_wm_text": float(r),
                        "roc_auc_score": float(roc_auc),
                        "avg_T": float(np.mean(len_noisy_data)),
                        "min_T": float(np.min(len_noisy_data)),
                        "max_T": float(np.max(len_noisy_data)),
                    }
                )

            with open("roc_auc_cp.json", "w") as fp:
                json.dump(roc_auc_scores, fp)

        if args.llm_attack:
            llm_results = list()

            for item in news:
                paraphrased_text = list()
                for prompt in prompts:
                    paraphrased_text.append(llm_paraphrasing(prompt, item["wm_text"]))

                paraphrased_data_results = [detect(text, args, device=device, tokenizer=tokenizer)[0] for text in paraphrased_text]
                z_score_cp = list()
                for p_text in paraphrased_data_results:
                    for subitem in p_text:
                        if subitem[0] == "z-score":
                            z_score_cp.append(float(subitem[1]))

                z_scores_baseline = [text["with_wm_scores"] for text in news]
                roc_auc = get_roc_auc_scores(z_score_cp, z_scores_baseline)

                len_noisy_data = [len(text) for text in paraphrased_text]

                llm_results.append(
                    {
                        "text": item["wm_text"],
                        "paraphrased_text": paraphrased_text,
                        "roc_auc_score": float(roc_auc),
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
