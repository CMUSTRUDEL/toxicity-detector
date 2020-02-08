# Toxicity Detection
These are installation instructions for the replication package for the ICSE NIER 2020 paper 
"Stress and Burnout in Open Source: Toward Finding, Understanding, and Mitigating Unhealthy Interactions" 
by Naveen Raman, Michelle Cao, Christian Kästner, Yulia Tsvetkov, and Bogdan Vasilescu
([preprint available here](toxicity.pdf)) 

# Data 
We have all the training data available from the paper in the data folder. 
All training data is in the data/training folder, and is in a .csv format. 
To train, use the train_comments file, which contains information about comments in addition to a label (toxicity) as y or n. 
To test the model, test against the data in labeled_test_comments.csv (the label is the column "toxicity"). 
To further test the model, we have a collection of 75K unlabeled issues randomly selected from Gtihub.
Because of file limitations, we only have the link to the repository. 
Run the classifier over the issues, and manually verify some of the issues predicted as toxic. 

We additionally have the data used to generate the plots relating to language, corporate status, and date based toxicity. 
These are all in csv format, and contain a list of links to Github issues. 

# Pretrained Model 

# Training the model 
To train your own model, use the training files in the training folder in data. 
For our model, we used an SVM classifier with perspective_score and stanford_polite, with C=.05, gamma=2, 
and the ratio between non-toxic and toxic to be 2. 
Alternatively, convert the train_issues.csv and train_comments.csv into MongoDB collections called naveen_issues and naveen_labeled_stanford_comments respectively. 
Then run the test_on function from get_results with your specified issue_collection, comment_collection, and output_file name. 
All the issues from the issue_collection will be classified by the classifier. 

# Plotting data
To reproduce the plots, first reclassify the data using get_results.py, and then plot the data using plot.py. 
For example, to reproduce the bar graph of language toxicity, run the classify_language function. This will store the results for each language in their own txt file. Next, run the plot_languages function, which will store the plots in plots/langs.png.  

# Getting Perspective Score 
To get the perspective score for a text, fill in the API key in perspective.py, and run the get_perspective_score function. 

 
