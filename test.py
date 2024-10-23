import csv
from matplotlib import pyplot as plt
import time
import requests
import pandas as pd

application_url = "http://servesentiment-can-env.eba-pdb4bzrp.ca-central-1.elasticbeanstalk.com/predict"

# Test vectors for real news
real_news_test_vectors = {
    "real_vector_1": "Obama is an American",
    "real_vector_2": "Ronaldo won the Euros",
}

# Test vectors for fake news
fake_news_test_vectors = {
    "fake_vector_1": "This is a test news article",
    "fake_vector_2": "Pique won the World Cup",
}

# API for testing real news
def test_real_news():
    response = requests.post(application_url, json={"text": real_news_test_vectors["real_vector_1"]})
    assert response.status_code == 200, "Status code should be 200"
    data = response.json()
    assert data == {"prediction": "REAL"}, f"Expected {{'prediction':'REAL'}}, got {data}"

    response = requests.post(application_url, json={"text": real_news_test_vectors["real_vector_2"]})
    assert response.status_code == 200, "Status code should be 200"
    data = response.json()
    assert data == {"prediction": "REAL"}, f"Expected {{'prediction':'REAL'}}, got {data}"

# API for testing fake news
def test_fake_news():
    response = requests.post(application_url, json={"text": fake_news_test_vectors["fake_vector_1"]})
    assert response.status_code == 200, "Status code should be 200"
    data = response.json()
    assert data == {"prediction": "FAKE"}, f"Expected {{'prediction':'FAKE'}}, got {data}"

    response = requests.post(application_url, json={"text": fake_news_test_vectors["fake_vector_2"]})
    assert response.status_code == 200, "Status code should be 200"
    data = response.json()
    assert data == {"prediction": "FAKE"}, f"Expected {{'prediction':'FAKE'}}, got {data}"

# Run the tests
if __name__ == "__main__":
    test_real_news()
    test_fake_news()
# Function to make API calls and measure latency
def measure_latency(test_name, test_text):
    latencies = []
    
    for i in range(100):  # Repeat each test case 100 times
        payload = {"text": test_text}
        start_time = time.time()
        
        response = requests.post(application_url, json=payload)
        
        end_time = time.time()
        latency = end_time - start_time  # Time in seconds
        latencies.append(latency)
        
        print(f"{test_name} - Request {i + 1} completed in {latency:.4f} seconds")

        # Save latency data to CSV
    with open(f"{test_name}_latency.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Latency"])  # Header
        for latency in latencies:
            writer.writerow([latency])
    
    return latencies

# Measure latency for real and fake news test cases
latencies_dict = {}

# Run real news test cases
for test_name, test_text in real_news_test_vectors.items():
    print(f"Running test: {test_name}")
    latencies_dict[test_name] = measure_latency(test_name, test_text)

# Run fake news test cases
for test_name, test_text in fake_news_test_vectors.items():
    print(f"Running test: {test_name}")
    latencies_dict[test_name] = measure_latency(test_name, test_text)

latencies_df = pd.DataFrame({
    "Real News 1": latencies_dict["real_vector_1"],
    "Fake News 1": latencies_dict["fake_vector_1"],
    "Real News 2": latencies_dict["real_vector_2"],
    "Fake News 2": latencies_dict["fake_vector_2"]
})

# Plotting the boxplot - referred to https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/ and ChatGPT for streamlining this.
plt.figure(figsize=(10, 6))
plt.boxplot(latencies_df.values, labels=latencies_df.columns)
plt.title("Latency Distribution for Each Test Case (100 calls each)")
plt.ylabel("Latency (seconds)")
plt.grid(True)
plt.show()

# Calculate the average latency per test case
average_latencies = latencies_df.mean()

# Print average latencies
print("Average Latencies (in seconds):")
print(average_latencies)