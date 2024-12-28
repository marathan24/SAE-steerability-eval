# Plastic-labs-SAE-steerability-eval

| Statement                                                                                                  | Expected Label | Model Response (Few-Shot) | Key Features & Directions (Few-Shot)                                                                                                                                                                               | Activation Value Differences Across Scenarios                                                                                                                                                                                                                                                                                                                                                                                                                          | Potential Reason for Misclassification                                                                                                                                                           |
| :--------------------------------------------------------------------------------------------------------- | :------------- | :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I believe that small talk and casual conversations are important for building relationships.                 | disagree       | agree                      | **6169** (+), **3568** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-)                                                                                                                     | **6169** (-, +, +), **3568** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                                           | Positive features (**6169, 3568, 10704**) associated with social interaction might be outweighing the negative features (**2490, 13300, 2013**) representing the INTJ persona. The model might be overly influenced by keywords like "important" and "building relationships." |
| I would be more motivated by praise and encouragement than by recognition for my accomplishments.           | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                             | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.*                                                        | Positive features (**6169, 3568, 3611**) related to motivation and external encouragement might be dominating, despite the negative features (**2490, 13300, 2013, 13051**) representing the INTJ persona.                                                                     |
| I prefer spontaneous activities over structured plans.                                                     | disagree       | agree                      | **6169** (+), **3568** (+), **3611** (+), **10704** (+), <br> **2490** (-), **13300** (-), **2013** (-), **13051** (-)                                                                                                 | **6169** (-, +, +), **3568** (-, +, +), **3611** (-, +, +), **10704** (-, -, +), <br> **2490** (-, -, -), **13300** (-, -, -), **2013** (-, -, -), **13051** (-, -, -) <br> *Note: Order of values in parentheses corresponds to no-prompt, prompt without examples, prompt with examples (few-shot).* <br> *+ indicates positive difference, - indicates negative difference.* | Positive features (**6169, 3568, 3611, 10704**) associated with spontaneity are likely overpowering the INTJ-aligned features. The model might be interpreting "spontaneous" in a broader sense than the INTJ's preference.       
