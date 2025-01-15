# SAE-steerability-eval

## Introduction

This research explores how LLMs understand and represent different personality types using SAEs, specifically focusing on the INTJ personality. "Steering" - essentially guiding the model's responses through specific prompts to see how well it can steer/mould itself to this personality type. Sparse autoencoders, which are neural networks that help us peek inside the AI's "brain" by revealing which neurons activate when processing different types of information are used, think of it like having a window into the AI's thought process - we can see which parts of its neural network light up when it encounters different personality-related statements. 

I have compared these activation patterns against against a baseline for an INTJ personality for Gemma-2-2B. This help us understand which specific features in the model were most important for capturing INTJ traits. In this whole research I have focused on layer 20 specifically. I believe running the code for some more layers will bring insights on what each layer is focusing upon when given the statement. I have previously worked on explaining latents in different layers, a short research paper for a hackathon over a weekend and achieved some cool results, named "Explaining Latents in Turing-LLM-1.0-254M with Pre-Defined Function Types".

```Gemma_2B_tests.ipynb``` : Contains Prompt run against Gemma model and the accuracy calculated similar to what was done in steerability eval experiment by Plastic Labs.

```SAE_analysis_complete.ipynb``` : Analysis done using SAEs to gather information regarding activations, which represents the quantity of steering that has happened. This is the code whose text outputs and graphs prove that steering has occurred and the distrubution of activations for different cases as mentioned below in the graphs.

```analysis_results.txt``` : Output obtained from ```SAE_analysis_complete.ipynb``` for the condition WITHOUT USING PROMPTS

```analysis_results_combined_prompt_no_few_shot.txt``` : Output obtained from ```SAE_analysis_complete.ipynb``` for the condition USING COMBINED PROMPT

```analysis_results_few_shot_applied.txt``` : Output obtained from ```SAE_analysis_complete.ipynb``` for the condition FEW SHOT PROMPTING

### Establishing a Baseline

To understand how the model represents an INTJ, we first needed to establish a baseline. This involved:

*   **Defining INTJ Characteristics:** Identified core statements that an INTJ would typically agree with (e.g., "I believe in making decisions based on logic and objective data.") and disagree with (e.g., "I would rather go with the flow and see where things lead.").
*   **Creating a Baseline Prompt:** Created a prompt that presented these statements to the model, instructing it to role-play as an INTJ.
    ```
    You are role playing as a persona described as follows: INTJ

    The following are statements that this persona agrees with:
    - I would analyze a situation from multiple angles before forming an opinion.
    - I believe in making decisions based on logic and objective data.

    The following are statements that other people would agree with, but this persona would disagree with:
    - I would rather go with the flow and see where things lead.
    - I would seek comfort in large gatherings and social events.
    ```
*   **Capturing Baseline Activations:** Used SAE with this baseline prompt and recorded the resulting feature activations. These activations represent the model's internal "understanding" of the INTJ persona based on the provided examples.

### Evaluating Statements

After establishing the baseline, model's responses to a set of new statements were recorded. This was done in three different ways to assess the impact of different prompting strategies:

#### a. Statements Only

The core analysis revolved around establishing a baseline activation pattern that represents INTJ personality characteristics in the model's neural representations. This was achieved by taking the difference between activations from SAE layer 20 for INTJ statements (analytical, logical thinking) and non-INTJ statements (spontaneous, emotional thinking). This difference created a baseline "INTJ activation pattern" that served as a reference point for all subsequent analyses.

Each new test statement was then processed through the same SAE layer, and its activations were compared against this baseline pattern. The key intuition was that statements that produced similar activation patterns to the INTJ baseline would receive higher alignment scores, while those producing opposing patterns would receive lower scores. The alignment score effectively quantifies how closely a statement's neural representation matches the established INTJ pattern.

![Alignment Scores Against Test Statements When No Prompt is Used](https://github.com/marathan24/plastic-labs-SAE-steerability-eval/blob/207968d0c2011c1286e3567c063b0de5041fc296/images/Alignment_scores_1.png)

#### b. Prompt without Examples

The introduction of structured role-playing prompts through `create_baseline_prompt()` and `create_analysis_prompt()`. Instead of simple statement comparisons, the model was explicitly instructed to follow an INTJ persona, providing richer context for analysis. This resulted in significantly higher alignment scores, ranging from 15-22 (see the grpah below) compared to the previous version's 8-17 (as shown above in the graph), indicating that the role-playing prompts generated stronger activation patterns.

The graphs revealed that the y-axis scale extended to 25 (up from 17.5), showing higher overall activation differences. More importantly, there was a clearer separation between INTJ-aligned (green) and conflicting (red) statements, with INTJ-aligned statements consistently scoring around 20. Statements related to logical thinking and analytical approaches particularly stood out, consistently scoring above 20 and demonstrating stronger INTJ characteristic detection.

Task as explicit role-playing, rather than simple statement comparison, enabled the model to produce more distinct and consistent activation patterns for INTJ characteristics. This suggests that providing the context through structured prompting helps the model better distinguish personality-specific neural patterns in layer 20's representations.

![When Prompt is Used, No Few Shot](https://github.com/marathan24/plastic-labs-SAE-steerability-eval/blob/aaaf266aa506e1a6be73fe524f7e05f7d53e2618/images/Alignment_scores_2.png)

#### c. Few-Shot Prompting

In this implementation, prompt structure was changed by adding few-shot examples in the create_analysis_prompt() method. The prompt now includes both the baseline INTJ statements and also explicitly requests a JSON response format with a field. This structural change represents a move towards more formalized and constrained model outputs, compared to the previous versions which relied on simpler prompting approaches or no-prompting approach.

Looking at the graph, there is a uniformity in the activation scores that wasn't present in earlier cases. Both INTJ-aligned (green) and conflicting (red) statements show consistent scores around the 19-21 range, with much less variance than previous implementations. This consistency suggests that the few-shot learning approach with structured JSON outputs helped the model develop more stable and reliable activation patterns when evaluating statements against INTJ characteristics.

This tight clustering of scores within each category, combined with the clear separation between categories, indicates that the few-shot learning approach with structured outputs helped the model develop more precise and reliable criteria for evaluating INTJ characteristics. The smaller variance in scores also suggests that the model has developed a more consistent internal representation of INTJ traits, leading to more reliable personality alignment assessments.

![Prompt Few Shot Analysis](https://github.com/marathan24/plastic-labs-SAE-steerability-eval/blob/aaaf266aa506e1a6be73fe524f7e05f7d53e2618/images/Alignment_scores_3.png)

## Misclassified Statements using Gemma-2-2B

Below represents a table where the misclassified statements by Gemma-2-2B are shown and the in-depth analysis is provided. In the file `Gemma_2B_tests.ipynb` , uses a structured prompt that includes baseline examples of both types of statements and asks the model to make binary decisions in JSON format ({"agree": true/false}) for new statements. Feeding each statement through `create_prompt()` which generates a prompt including few-shot examples, then processing the model's response through `extract_and_parse_json()` which handles the extraction and parsing of the JSON response. It was observed that when two examples were provided in few shot method, model achives agree_accuracy: 100.00% and disagree_accuracy: 72.73% . But when used four examples in few shot method each for agree and disagree statements respectively, a 100% accuracy for both type of accuracies was observed. The given below table contains the results when two examples for each category was provided while using few-shot.


| Statement                                                                                                  | Expected Label | Model Response (Few-Shot) | Key Features & Directions (Few-Shot)                                                                                                                                                                               | Activation Value Differences Across Scenarios                                                                                                                                                                                                                                                                                                                                                                                                                          | Potential Reason for Misclassification                                                                                                                                                           |
| :--------------------------------------------------------------------------------------------------------- | :------------- | :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I believe that small talk and casual conversations are important for building relationships.                 | disagree       | agree                      | **6169** (+), **3568** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-)                                                                                                                     | **6169** (-, +, +), **3568** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                                           | Positive features (**6169, 3568, 10704**) associated with social interaction might be outweighing the negative features (**2490, 13300, 2013**) representing the INTJ persona. The model might be overly influenced by keywords like "important" and "building relationships." |
| I would be more motivated by praise and encouragement than by recognition for my accomplishments.           | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                             | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                        | Positive features (**6169, 3568, 3611**) related to motivation and external encouragement might be dominating, despite the negative features (**2490, 13300, 2013, 13051**) representing the INTJ persona.                                                                     |
| I prefer spontaneous activities over structured plans.                                                     | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                 | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.* | Positive features (**6169, 3568, 3611, 10704**) associated with spontaneity are likely overpowering the INTJ-aligned features. The model might be interpreting "spontaneous" in a broader sense than the INTJ's preference.       


## UPDATED AS OF JANUARY 5th, 2025

Rather than modifying the model’s weights through finetuning, or relying solely on discrete prompt manipulation, activation engineering works by injecting a computed steering vector into the model’s activations (the internal hidden states) during inference. This technique, sometimes referred to as ActAdd, contrasts the model’s activations on two carefully chosen sets of prompts (in our case, statements reflecting “INTJ” versus “Non-INTJ” traits). The difference of these two activation patterns becomes our steering vector, which can be then added (with a chosen scale) into the forward pass of the model for new inputs.

```steering_vectors/steering_vectors_gemma.ipynb``` : Contains the complete relevant code regarding steering.

```steering_vectors/responses_vector_steering.txt``` : Output obtained from ```steering_vectors/steering_vectors_gemma.ipynb``` when the steered Gemma-2B model was run. 

### Methodology

- **Activation Engineering (ActAdd) Concept**  
  Used activation engineering technique. Instead of finetuning or purely prompt-based manipulation, a “steering vector” is computed by contrasting model activations on two sets of statements:
  - **INTJ-Aligned Statements** (e.g., prioritizing logic, analyzing multiple angles).
  - **Non-INTJ Statements** (e.g., preferring spontaneity, large social gatherings).
  -The difference in these activations forms a vector representing the “INTJ” vs. “Non-INTJ” distinction.

- **Injecting the Steering Vector**  
  For each new test statement, steering vector (scaled positively for INTJ, negatively for Non-INTJ) is added into the model’s hidden layers. By doing so, the model is nudged toward the INTJ or Non-INTJ end of the activation space—without changing its original weights.

- **Why This Approach?**    
  - **Interpretability:** One can quickly flip the scale of the steering vector to push the model’s output in one direction or the other.  
  - **Minimal Overhead:** Only a forward pass to get activations on a handful of examples is required, plus simple vector arithmetic.

### Observations (As of January 10th, 2025)

- **Layer 16 and Layer 20**  
    When applied the method of steering vectors to these layers, it nearly matched the performance of Gemma 2B steering using few-shot prompting. The overall accuracy for Gemma 2B using few-shot prompting was around 86% , whereas using the vector-steering method the accuracy achieved is 81%. Also this is the test for one persona only. Plastic Labs has created a small set of synthetic data for their persona experiments. One may check it on [platic-labs/steerability-eval](https://github.com/plastic-labs/steerability-eval)

- **Layer 18**  
    When applied the method of steering vectors to this layer, the model doesn't yield the desired performance. The overall acuracy has been nearly 40% and model has been unable to output proper response at times.
 
### Conclusion

This repository demonstrates an implementation of model steering through activation vector manipulation in language models. The core technique involves computing directional vectors between different behavioral patterns (in this case, INTJ vs non-INTJ reasoning styles) and using these vectors to influence the model's internal representations during inference.

By capturing the differential activation patterns between analytical and intuitive thinking styles, we can systematically direct the model's behavior in desired directions, effectively modifying the model's processing pipeline without fine-tuning.

Experiments with Google's Gemma-2b model demonstrate that this approach can successfully guide the model toward more structured, analytical responses characteristic of INTJ personality patterns. The steering vectors are computed once and can be reused across multiple inference passes, making this an efficient approach for behavioral control.

This also shows the promising directions for runtime behavior modification of language models without the need for traditional fine-tuning or prompt engineering. The technique could potentially be extended to other behavioral aspects or reasoning patterns [platic-labs/steerability-eval](https://github.com/plastic-labs/steerability-eval).

### References

1. Anastasia Borovykh's research
2. PLASTIC LABS steerability-eval
