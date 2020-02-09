# Toxicity Detection
These are installation instructions for the replication package for the ICSE NIER 2020 paper 
"Stress and Burnout in Open Source: Toward Finding, Understanding, and Mitigating Unhealthy Interactions" 
by Naveen Raman, Michelle Cao, Christian Kästner, Yulia Tsvetkov, and Bogdan Vasilescu
([preprint available here](toxicity.pdf)) 

# Data 
We have all the training data available from the paper in the data folder. 
All training data is in the data/training folder, and is in a .csv format. 
To train, use the train_comments file, which contains information about comments in addition to a label (toxic) as y or n. 
To test the model, test against the data in labeled_test_issues.csv (the label is the column "toxic"). 
To further test the model, we have a collection of 75K unlabeled issues randomly selected from Gtihub.
Because of file limitations, we only have the link to the repository. 
Run the classifier over the issues, and manually verify some of the issues predicted as toxic. 

We additionally have the data used to generate the plots relating to language, corporate status, and date based toxicity. 
These are all in csv format, and contain a list of links to Github issues. 

Sample json entry for Mongo 
```
{'_id':'Apktool/iBotPeaches/1707/287520345',
'polarity':-0.10625,
'perspective_score':0.08018797,
'owner':iBotPeaches,
'repo':'Apktool',
'num_reference':0,
'text':'I am using apktool v 2.3.1. on Kali Linux. I get this error. Yesterday I have tried and it worked, today I get this error with any APK, tried at least 10. "" Using APK template Whatsappo.apk No platform was selected, choosing MsfModulePlatformAndroid from the payload No Arch selected, selecting Arch dalvik from the payload [] Creating signing key and keystore.. [] Decompiling original APK.. [] Decompiling payload APK.. [] Locating hook point.. [] Adding payload as package com.whatsapp.zumvr [] Loading /tmp/d20180110-6408-1ej4cd4/original/smali/com/whatsapp/AppShell.smali and injecting payload.. [] Poisoning the manifest with meterpreter permissions.. [] Adding &lt;uses-permission androidname=""android.permission.READ_CALL_LOG""/&gt; [] Adding &lt;uses-permission androidname=""android.permission.SET_WALLPAPER""/&gt; [] Adding &lt;uses-permission androidname=""android.permission.READ_SMS""/&gt; [] Adding &lt;uses-permission androidname=""android.permission.CALL_PHONE""/&gt; [] Adding &lt;uses-permission androidname=""android.permission.WRITE_CALL_LOG""/&gt; [*] Rebuilding Whatsappo.apk with meterpreter injection as /tmp/d20180110-6408-1ej4cd4/output.apk Error Unable to rebuild apk with apktool "" "',
'num_url':0,
'stanford_polite':0.4867653863696374,
'subjectivity':0.6625,
'num_emoji':2,
'num_mention':0,
'nltk_score':'-0.9127,
'toxicity':0}
```

Though our model only uses the text, stanford_polite, and perspective_score, one could build a model using the various other features. 

# Pretrained Model 
We have a pretrained SVM model available in the src folder. 
The model is based off the sklearn SVM classifier object, and has two features, perspective_score and stanford_polite (both of which are numbers between 0-1). 
To filter words by software specific words, train a model from scratch as described below. 
To load the pretrained model, use 
```
import pickle
a = pickle.load(open("pretrained_model.p","rb"))
# Perspective Score, Stanford Polite 
a.predict([[0.1,0.2]])
a.predict([[1,1]])
```

# Training the model 
To train your own model, use the training files in the training folder in data. 
For our model, we used an SVM classifier with perspective_score and stanford_polite, with C=.05, gamma=2, 
and the ratio between non-toxic and toxic to be 2. 
Alternatively, convert the train_issues.csv and train_comments.csv into MongoDB collections called naveen_issues and naveen_labeled_stanford_comments respectively. 
Then run the test_on function from get_results with your specified issue_collection, comment_collection, and output_file name. 

```
import get_results 
get_results.test_on("new_issues","new_comments","output_file")
```
All the issues from the issue_collection will be classified by the classifier, and stored in the output_file.  

# Plotting data
To reproduce the plots, first reclassify the data using get_results.py, and then plot the data using plot.py. 
For example, to reproduce the bar graph of language toxicity, run the classify_language function. 
This will store the results for each language in their own txt file. Next, run the plot_languages function, which will store the plots in plots/langs.png.  
For example, to reproduce the bar graph of language toxicity, run the classify_language function. This will store the results for each language in their own txt file. Next, run the plot_languages function, which will store the plots in plots/langs.png.  
```
import get_results 
import plot 
get_results.classify_language()
plot.plot_languages()
```

# Getting Perspective Score 
To get the perspective score for a text, fill in the API key in perspective.py, and run the get_perspective_score function. 
```
import perspective
perspective.API_KEY = ""
perspective.get_perspective_score("Sentence")
```
 
