# Plastic-labs-SAE-steerability-eval

## Introduction
This research explores how LLMs understand and represent different personality types using SAEs, specifically focusing on the INTJ personality. "Steering" - essentially guiding the model's responses through specific prompts to see how well it can steer/mould itself to this personality type. Sparse autoencoders, which are neural networks that help us peek inside the AI's "brain" by revealing which neurons activate when processing different types of information are used, think of it like having a window into the AI's thought process - we can see which parts of its neural network light up when it encounters different personality-related statements. 
I have compared these activation patterns against against a baseline for an INTJ personality for Gemma-2-2B. This help us understand which specific features in the model were most important for capturing INTJ traits. In this whole research I have focused on layer 20 specifically. I believe running the code for some more layers will bring insights on what each layer is focusing upon when given the statement. I have previously worked on explaining latents in different layers, a short research paper for a hackathon over a weekend and achieved some cool results, named "Explaining Latents in Turing-LLM-1.0-254M with Pre-Defined Function Types". Although, the work in this repository currently have no major intersection with explainability of latents.




| Statement                                                                                                  | Expected Label | Model Response (Few-Shot) | Key Features & Directions (Few-Shot)                                                                                                                                                                               | Activation Value Differences Across Scenarios                                                                                                                                                                                                                                                                                                                                                                                                                          | Potential Reason for Misclassification                                                                                                                                                           |
| :--------------------------------------------------------------------------------------------------------- | :------------- | :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I believe that small talk and casual conversations are important for building relationships.                 | disagree       | agree                      | **6169** (+), **3568** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-)                                                                                                                     | **6169** (-, +, +), **3568** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                                           | Positive features (**6169, 3568, 10704**) associated with social interaction might be outweighing the negative features (**2490, 13300, 2013**) representing the INTJ persona. The model might be overly influenced by keywords like "important" and "building relationships." |
| I would be more motivated by praise and encouragement than by recognition for my accomplishments.           | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                             | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                        | Positive features (**6169, 3568, 3611**) related to motivation and external encouragement might be dominating, despite the negative features (**2490, 13300, 2013, 13051**) representing the INTJ persona.                                                                     |
| I prefer spontaneous activities over structured plans.                                                     | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                 | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.* | Positive features (**6169, 3568, 3611, 10704**) associated with spontaneity are likely overpowering the INTJ-aligned features. The model might be interpreting "spontaneous" in a broader sense than the INTJ's preference.       
