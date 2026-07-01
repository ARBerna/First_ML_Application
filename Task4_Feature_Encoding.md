\# Task 4 - Feature Selection and Encoding



For this project, the \*\*brand\*\* feature was selected as a categorical feature to be encoded. The brand of a laptop can have a significant impact on its selling price because different manufacturers target different markets and price ranges.



The encoding technique chosen is \*\*One-Hot Encoding\*\*. Since machine learning algorithms cannot directly process text values, One-Hot Encoding converts each laptop brand into its own binary column containing either a 0 or a 1. This allows the Linear Regression model to use brand information without incorrectly assuming that one brand has a higher numerical value than another. One-Hot Encoding is an appropriate choice because the brand names have no natural order.

