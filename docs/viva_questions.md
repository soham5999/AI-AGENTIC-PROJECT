# Viva Questions and Answers

1. **What is an AI agent?** A system that observes inputs, uses tools or reasoning, takes actions and may retain memory toward a goal.
2. **What makes this project agentic?** It orchestrates validation, prediction, diagnosis, recommendation, storage and alert tools in sequence.
3. **AI vs ML vs Agentic AI?** AI is the broad field; ML learns patterns from data; agentic AI coordinates tools/actions toward an objective.
4. **Why predictive maintenance?** It can prioritize inspection before a service issue becomes disruptive.
5. **Why classification?** The target has two classes: maintenance predicted required or not currently required.
6. **What is SVM?** A classifier that finds a separating boundary with maximum margin, optionally using kernels.
7. **What is Random Forest?** An ensemble of randomized decision trees whose outputs are aggregated.
8. **What is Logistic Regression?** A linear probabilistic classifier using a logistic function for class probability.
9. **What is Gradient Boosting?** Sequential weak learners that correct previous errors.
10. **Why compare models?** Different assumptions and capacity may produce different generalization performance.
11. **Accuracy vs precision?** Accuracy is all correct predictions; precision is the fraction of predicted positives that are truly positive.
12. **Recall vs precision?** Recall measures how many actual positives were found; precision measures trust in positive alerts.
13. **What is F1?** The harmonic mean of precision and recall.
14. **What is a confusion matrix?** A table of true/false positive and negative predictions.
15. **What is overfitting?** Learning training-specific noise rather than general patterns.
16. **Why stratify the split?** To preserve class proportions in train and test sets.
17. **What is feature importance?** A model-associated measure of contribution or usefulness, not proof of causality.
18. **What is SQLite?** A serverless file-based relational database.
19. **Why does the agent need memory?** To retrieve prior vehicle analyses and support history-aware review.
20. **What are tools?** Small functions that perform defined operations such as prediction or storage.
21. **What is human-in-the-loop?** A qualified person reviews and makes the final decision.
22. **What is the LLM role?** Optional natural-language explanation of structured outputs.
23. **Why should the LLM not predict maintenance directly?** The tested ML pipeline provides reproducible structured prediction and the LLM could hallucinate.
24. **What if the API key is missing?** The deterministic Python agent still works; only explanation is unavailable.
25. **Why use synthetic data?** It permits reproducible learning without exposing proprietary vehicle data, but limits realism.
26. **Why save the model?** To reuse the trained pipeline consistently in the application.
27. **Why use probability?** It provides a continuous risk score for priority rules.
28. **What is data validation?** Checking values and types against acceptable bounds before inference.
29. **Why is calibration important in deployment?** A probability should correspond meaningfully to observed frequencies.
30. **Can high risk prove failure?** No. It only triggers a possible concern and inspection recommendation.
31. **What is SQLite memory storing?** ID, time, parameters, risk, prediction, diagnosis, recommendation and priority.
32. **What is a false negative?** Maintenance is actually needed but the model predicts it is not.
33. **What is a false positive?** The model recommends maintenance when it may not be needed.
34. **Why can recall matter?** Missing a maintenance-needed case may be more serious than an extra inspection recommendation.
35. **What are project limitations?** Synthetic data, no physical inspection, no temporal sensor stream and no deployment validation.
36. **What is future scope?** Real validated data, calibration, drift monitoring, technician feedback and safe notification integration.
37. **Why use a pipeline?** It keeps preprocessing and inference together, reducing train-serving mismatch.
38. **What is reproducibility?** Re-running with the same seed and code produces comparable generated data and training procedure.
39. **Why are alerts simulated?** The academic prototype should not send real messages without explicit configuration and consent.
40. **What is the final safety principle?** AI output is decision support; a qualified technician verifies all maintenance decisions.
