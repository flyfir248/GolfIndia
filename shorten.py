import pandas as pd

# Load the large CSV file
large_csv_path = 'C:/Users/anoop/PycharmProjects/MITProject/Capstone_AnoopJohny_StatusUpdate/Dataanddashboard/glassdoor-job-reviews/glassdoor_reviews.csv'
# Specify the path to save the shortened CSV file
shortened_csv_path = 'C:/Users/anoop/PycharmProjects/tester/DataShortened/glassdoor_shortened.csv'

# Read the large CSV file, limiting to the first 100,000 rows
df = pd.read_csv(large_csv_path, nrows=5000)

# Save the shortened dataframe to a new CSV file
df.to_csv(shortened_csv_path, index=False)

print(f'Successfully saved the first 100,000 rows to {shortened_csv_path}')
