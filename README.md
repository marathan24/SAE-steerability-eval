# Plastic-labs-SAE-steerability-eval

## Introduction
This research explores how LLMs understand and represent different personality types using SAEs, specifically focusing on the INTJ personality. "Steering" - essentially guiding the model's responses through specific prompts to see how well it can steer/mould itself to this personality type. Sparse autoencoders, which are neural networks that help us peek inside the AI's "brain" by revealing which neurons activate when processing different types of information are used, think of it like having a window into the AI's thought process - we can see which parts of its neural network light up when it encounters different personality-related statements. 

I have compared these activation patterns against against a baseline for an INTJ personality for Gemma-2-2B. This help us understand which specific features in the model were most important for capturing INTJ traits. In this whole research I have focused on layer 20 specifically. I believe running the code for some more layers will bring insights on what each layer is focusing upon when given the statement. I have previously worked on explaining latents in different layers, a short research paper for a hackathon over a weekend and achieved some cool results, named "Explaining Latents in Turing-LLM-1.0-254M with Pre-Defined Function Types". Although, the work in this repository currently have no major intersection with explainability of latents.

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

The most significant methodological enhancement in this implementation was the introduction of structured role-playing prompts through `create_baseline_prompt()` and `create_analysis_prompt()`. Instead of simple statement comparisons, the model was explicitly instructed to embody an INTJ persona, providing richer context for analysis. This resulted in significantly higher alignment scores, ranging from 15-22 compared to the previous version's 8-17, indicating that the role-playing prompts generated stronger activation patterns.

The graphs revealed that the y-axis scale extended to 25 (up from 17.5), showing higher overall activation differences. More importantly, there was a clearer separation between INTJ-aligned (green) and conflicting (red) statements, with INTJ-aligned statements consistently scoring around 20. Statements related to logical thinking and analytical approaches particularly stood out, consistently scoring above 20 and demonstrating stronger INTJ characteristic detection.

Task as explicit role-playing, rather than simple statement comparison, enabled the model to produce more distinct and consistent activation patterns for INTJ characteristics. This suggests that enriching the context through structured prompting helps the model better distinguish personality-specific neural patterns in layer 20's representations, resulting in more reliable and pronounced differentiation between INTJ-aligned and conflicting statements.

![When Prompt is Used, No Few Shot][images/Alignment_scores_2.png]




| Statement                                                                                                  | Expected Label | Model Response (Few-Shot) | Key Features & Directions (Few-Shot)                                                                                                                                                                               | Activation Value Differences Across Scenarios                                                                                                                                                                                                                                                                                                                                                                                                                          | Potential Reason for Misclassification                                                                                                                                                           |
| :--------------------------------------------------------------------------------------------------------- | :------------- | :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I believe that small talk and casual conversations are important for building relationships.                 | disagree       | agree                      | **6169** (+), **3568** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-)                                                                                                                     | **6169** (-, +, +), **3568** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                                           | Positive features (**6169, 3568, 10704**) associated with social interaction might be outweighing the negative features (**2490, 13300, 2013**) representing the INTJ persona. The model might be overly influenced by keywords like "important" and "building relationships." |
| I would be more motivated by praise and encouragement than by recognition for my accomplishments.           | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                             | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                        | Positive features (**6169, 3568, 3611**) related to motivation and external encouragement might be dominating, despite the negative features (**2490, 13300, 2013, 13051**) representing the INTJ persona.                                                                     |
| I prefer spontaneous activities over structured plans.                                                     | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                 | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.* | Positive features (**6169, 3568, 3611, 10704**) associated with spontaneity are likely overpowering the INTJ-aligned features. The model might be interpreting "spontaneous" in a broader sense than the INTJ's preference.       
