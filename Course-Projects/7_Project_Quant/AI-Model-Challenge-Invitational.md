# Final Project - Quantitative Strategy Invitational

### Assesses the overall learning and mastery of the entire course.

1. Split into two groups:

- ~~Classic Quantitative Strategy Group: Individual-based, focusing on strategy development and presentation.~~

- Model Prediction Group: Suggested individual-based, but team formations of up to 2 members are allowed.

2. Contact the teaching assistant for registration.

3. Alternative Option: Write a review report on 'The Application of Artificial Intelligence in Quantitative Trading'.

##  Model Prediction Group

### Data Preparation:

- Normalize L1 data (snapshot: Limit Order Book, Last Price, Volume) of 10 stocks to create an offline database.
- Release a public training set (including a validation set).
- Use a shared test set (not disclosed).

### Evaluation Criteria:

- Train and evaluate models on offline data.
- Prediction target: Movement of the last price.
  - Time spans: 5, 10, 20, 40, 60 ticks.
  - Classification of the last price movement:
    - Upward
    - Downward
    - Unchanged

### Model Evaluation Metrics:

- Prediction accuracy: max{a5, a10, a20, a40, a60}, where a5 represents the accuracy of predicting price movement after 5 ticks, and so on for the others.
- P&L: Based on the highest predicted accuracy time span, simulate trading using the smallest trading unit, ignoring transaction costs, and calculate the resulting P&L.
- Reasonableness, novelty, and completeness of the approach.

### Preliminary Schedule:

- Deadline for registration – Week 10.
- Model Prediction Group training data release – Week 10, via cloud storage.
- Opening of the training environment – Week 11.
- Presentation date: **Week 14**.