# Football_Analysis_NLP
Identification of Undervalued Football Players in Transfer Markets through NLP- Based News Analysis
## Project overview
This project was a deep dive into using NLP and machine learning to identify undervalued football players in the transfer market. Below is a detailed breakdown of everything that was done, including the steps in the code, the analysis, and the outputs:

### Data Source
These datasets were provided by Typewind Ltd and were stored in Google Drive for processing in Google Colab.

### 1. Data Collection and Setup
- Datasets Used:
   - News Dataset: 212,853 football-related news articles, containing raw content, titles, dates, and other metadata.
   - Transfer Datasets: Three datasets covering player details (20,415 players), transfer history (151,436 records), and market value development (280,334 records).

- Environment Setup:
  - Used Google Colab with Google Drive integration to manage large datasets.
  - Installed dependencies like pandas, numpy, nltk, spacy, scikit-learn, transformers, beautifulsoup4, matplotlib, seaborn, and swifter.
  - Downloaded spaCy’s en_core_web_trf model for entity extraction and NLTK resources for text processing.

### 2. Data Preprocessing
- News Data Cleaning:
   - Processed 212,853 news articles by removing HTML tags, special characters, and extra spaces using BeautifulSoup and regex.
   - Filtered articles to those under 2,000 characters to focus on concise, relevant content, reducing the dataset to 63,328 articles.
   - Dropped rows with missing processed_text, ensuring data quality.
- Player Name Extraction:
  - Extracted unique player names from the transfer dataset (19,784 players).
  - Matched player names in news articles using a case-insensitive algorithm, identifying 56,736 articles with player matches.
  - Kept only articles with matched players, resulting in a final news dataset of 56,736 rows.
- Transfer Data Cleaning:
   - Merged three transfer datasets (player details, transfer history, and market value development) into a unified DataFrame.
   - Kept the latest transfer per player and the highest market value per player.
   - Cleaned market value and transfer fee columns by converting strings (e.g., "€65m") to numerical values (e.g., 65,000,000), handling missing values, and dropping negative or invalid entries.
   - Final transfer dataset: 20,415 players with cleaned market values and transfer fees.
 
### 3. NLP and Sentiment Analysis
- Sentiment Analysis:
   - Used the finiteautomata/bertweet-base-sentiment-analysis model to classify news sentiment as Positive (POS), Negative (NEG), or Neutral (NEU).
   - Processed a subset of 5,000 articles in batches (to avoid Colab crashes), saving partial results to Google Drive.
   -  Results: 3,556 NEU, 1,206 POS, and 238 NEG articles, showing that most news is neutral, but positive sentiment is significant for identifying potential.
- Entity Extraction:
   - Applied spaCy’s en_core_web_trf model to extract player names and clubs from news titles.
   - Handled missing values by returning empty lists for NaN entries, ensuring robust processing.
   - Example output: For a title like "Varane named Man United captain vs Lyon," extracted entities were player_entities: ["Varane", "Lyon", "Maguire"] and club_entities: ["Man United"].

 ### 4. Data Merging and Feature Engineering
 - Merged Datasets:
    - Filtered transfer history for 2022–2024, resulting in 38,419 records.
    - Merged news data with transfer history and market value data using player IDs, focusing on players with both news and transfer records.
    - Final merged dataset: 219 players with news sentiment, transfer fees, market values, and club information.
- Feature Engineering:
   - Encoded categorical variables:
      - Sentiment (POS, NEG, NEU) using LabelEncoder.
      - Clubs (to_clubName) using LabelEncoder, handling missing values by filling with "Unknown".
   - Extracted the transfer year from the date column.
   - Features for modeling: market value (mw), sentiment (encoded), club (encoded), and transfer year.
   - Target: Transfer fee (Transfer Fee).

### 5. Predictive Modeling
- Models Trained:
   - Linear Regression: A baseline model to predict transfer fees.
   - Random Forest: A tree-based model to capture non-linear relationships.
   - Gradient Boosting: A boosting model to improve prediction accuracy.
   - Ensemble (VotingRegressor): Combined predictions from the above models for better performance.
- Training and Evaluation:
   - Split data into 80% training and 20% testing sets.
   - Evaluated models using RMSE (in €M) and R²:
      - Linear Regression: RMSE €14.54M, R² 0.3563 (best performer).
      - Random Forest: RMSE €16.38M, R² 0.1831.
      - Gradient Boosting: RMSE €15.75M, R² 0.2450.
      - Ensemble: RMSE €15.26M, R² 0.2908.
   - Linear Regression outperformed others, indicating that transfer fees have a relatively linear relationship with the features, though the R² suggests room for improvement.
- Feature Importance (Random Forest):
   - Market value (mw): 74.84% (dominant factor).
   - Club (encoded): 13.35%.
   - Transfer year: 6.66%.
   - Sentiment (encoded): 5.14%.
   - This shows that market value is the primary driver of transfer fees, but sentiment and club reputation also play a role.

### 6.  Recommendation System
- Criteria for Undervalued Players:
   - Positive sentiment in news articles (indicating potential for growth).
   - Market value less than the transfer fee (indicating overpayment).
   - Market value greater than €1M (to focus on significant players).
- Results:
   - Identified 17 undervalued players, ranked by overpayment (Transfer Fee - Market Value).
   - Top recommendations:
      - Antony (Man Utd): Overpayment €67M.
      - Marc Cucurella (Chelsea): Overpayment €40.3M.
      - Ferran Torres (Barcelona): Overpayment €20M.
   - Provided detailed summaries for each player, e.g., "Antony: Transferred to Man Utd for €95.0M in 2022, now valued at €28.0M. Positive news suggests untapped potential despite €67.0M overpayment."

### 7. Pattern Analysis
- Patterns Identified:
   - Positive Sentiment + Overpayment: 17 players with positive sentiment were overpaid, suggesting they are undervalued now and could be smart signings.
   - Negative Sentiment + Overpayment: 2 players with negative sentiment had market values less than their transfer fees, indicating risky investments.
   - Positive Sentiment + Good Value: 37 players with positive sentiment had market values exceeding their transfer fees, indicating successful signings.

 ### Visualizations
 - Bar Chart: Showed overpayment for undervalued players, with Antony having the highest overpayment (€67M).
 - Scatter Plot (Market Value vs. Transfer Fee): Highlighted undervalued players below the "Fair Value" line, with labels for each player.
 - Feature Importance Plot: Visualized the dominance of market value in predicting transfer fees.
 - Predicted vs. Actual Transfer Fees Plot: Compared predictions from all models against actual fees, showing Linear Regression’s alignment with the "Perfect Prediction" line.

### Conclusion
This project successfully developed a data-driven system to identify undervalued football players in the transfer market by integrating NLP and machine learning techniques. By analyzing 212,853 news articles and historical transfer data, with a specific focus on transfers from 2022 to 2024 to identify undervalued players, the system pinpointed 17 such players. Top recommendations included Antony (overpaid by €67M), Marc Cucurella (€40.3M), and Ferran Torres (€20M), all backed by positive news sentiment. Predictive modeling revealed that market value is the dominant factor in determining transfer fees (74.84% importance), with Linear Regression achieving the best performance (RMSE €14.54M, R² 0.3563). Pattern analysis uncovered actionable insights, such as the potential of players with positive sentiment and high overpayment, while visualizations effectively communicated these findings. Overall, this project provides football clubs with a strategic tool to optimize transfer decisions, potentially saving millions by targeting undervalued talent with high growth potential during the 2022–2024 period.

#### Lessons Learned from the Project
1. Importance of Data Quality and Preprocessing:
   - Cleaning and preprocessing large datasets (e.g., 212,853 news articles and transfer records) was critical to ensure accurate analysis. Issues like inconsistent formats (e.g., "€65m" vs. numerical values) and missing data required careful handling, highlighting the need for robust data pipelines in real-world projects.
2. Impact of Sentiment on Valuation:
   - News sentiment significantly influences player valuation, but its predictive power (5.14% feature importance) is secondary to market value (74.84%). This suggests that while public perception matters, financial metrics remain the primary driver in transfer markets, guiding the focus of future analyses.
3. Model Limitations and Feature Selection:
   - The moderate R² (0.3563 for Linear Regression) indicated that the models captured some, but not all, variance in transfer fees. This taught me that additional features (e.g., player performance metrics like goals or assists) could enhance predictive accuracy, emphasizing the importance of comprehensive feature engineering.
4. Value of Pattern Analysis:
   - Identifying patterns like "Positive Sentiment + Overpayment" (17 players) provided actionable insights for clubs, showing the power of combining NLP with transfer data to uncover strategic opportunities. This reinforced the value of exploratory data analysis in generating practical recommendations.
5. Visualization as a Communication Tool:
   - Visualizations (e.g., scatter plots, bar charts) were essential for communicating complex findings clearly, such as the overpayment of undervalued players. This underscored the importance of effective data storytelling to make insights accessible to stakeholders like football scouts.
6. Scalability and Real-World Application:
   - Working with large datasets in Google Colab highlighted the need for scalable solutions (e.g., batch processing for sentiment analysis). It also showed the real-world potential of the system to save clubs millions, teaching me how data science can directly impact business decisions in industries like sports.

### Future Improvements
The project has a solid foundation but can be enhanced in several ways to improve its accuracy, relevance, and usability for football clubs. Here are the planned future improvements:
1. Incorporate Player Performance Data:
   - Add metrics like goals, assists, minutes played, pass completion rates, and defensive stats to better capture a player’s on-field impact. This could improve the predictive accuracy of transfer fee models by providing a more holistic view of a player’s value beyond market value and sentiment.
2. Expand Sentiment Analysis:
   - Include social media sentiment from platforms like Twitter/X to capture real-time fan and media perceptions. This broader sentiment analysis can provide a more comprehensive understanding of a player’s public image, which often influences transfer decisions.
3. Real-Time Analysis:
   - Build a pipeline for real-time news scraping and analysis using APIs like NewsAPI. This would allow the system to provide up-to-date recommendations based on the latest news, making it more relevant for dynamic transfer windows.
4. User Interface Development:
   - Develop a user-friendly dashboard or web application for scouts to interact with the system, input player names, and receive real-time recommendations with visualizations. This would enhance the practical usability of the tool.
