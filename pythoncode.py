import json
import csv
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import math
import re
import pickle as p

# Cutoff coefficient for predicted probabilities.
# I usually try to pick a cutoff value that produces the same number of 
# predicted ones for the training set as there actually are.
cutoff_coeff = 0.4

# Number of words for our dictionaries to retain.
retain_words_number = 500;

# Timestamp for the first of December 2010; used in calculating RAOP age.
dec1_2010_timestamp = 1291161600

# Floats by which we scale request length and timestamp.
req_length_factor = float(1000)
timestamp_factor = float(10000000)

# Words we want to ignore for the purposes of compiling dictionaries of
# important keywords.
useless_words = ['I', 'to', 'and', 'a', 'the', 'my', 'for', 'of', 'in', 'it',\
    'have', 'pizza', 'me', 'is', 't', 'be', 'm', 'but', 'would', 'that', 'on',\
    'this', 'you', 'out', 's', 'with', 'so', 'get', 'can', 'been', 'we', 'just',\
    'some', 'i', 've', 'if', 'are', 'was', 'not', 'at','do','did','Not','All',\
    'will', 'as', 'am', 'all', 'from', 'money', 'up', 'pay', 'now', 'had', 'our',\
    'My', 'don', 'like', 'or','It','http','com','by','The','him','her','your'\
    'could', 'has', 'about', 'here', 'no', 'back', 'd', 'much','So','As','He',\
    'do', 'got', 'someone', 'when', 'an', 'one','isn','You','r','t','own','lot',\
    'us', 'anyone', 'any', 'next', 'there', 'If', 'he','A','too','bit','www',\
    'll','1','2','3','4','5','6','7','8','9','We','your','then','being','She',\
    'were','Just','didn','also','how','doesn','And','Im','im']
    
# Words expressing gratitude.
gratitude_words = ['Thanks', 'thanks', 'Thank', 'thank']

# Sensitive words which we expect to be highly likely to yield a 
# successful request.
sensitive_words = ['Pregnant','pregnant', 'flu', 'kidney', 'hospital',\
'died','hospital', 'suicide','institution', 'lonely','sober','sick','ill',\
'flu', 'baby','DOTA']

# Words that usually correspond to successful requests.
success_words = ['paycheck', 'check', 'recently','proof',\
'currently','wife','car','well','Friday', 'unemployed',\
'kids', 'hours', 'request','job','jobs', 'husband','gas','bills',\
'paycheck', 'paychecks','son', 'daughter','stamps','paid',\
'deposit','single','cash','Ramen', 'beans',\
'help','fiance','overdraft','dog','cat']

# Words that tend to appear in failed requests.
fail_words = ['friend', 'guys', 'friends','drunk','beer','nothing',\
'eating', 'awesome', 'reading','girlfriend','free','appreciated',\
'moved', 'won', 'afford','best','hoping','happy','Hi','story','starving',
'birthday','lunch','brother','cheese','full','party','roomies','laws',\
'happy','pothead','16','17','18','fucking','rage','repost','Canuck']

# Function that loads our desired columns from the given data set, where
# the data set is Train or Test.
def load_columns(data):
    
    data_len = len(data)
    
    #karma
    karma=[]
    for i in range(0, data_len):
        karma.append(data[i]['requester_upvotes_minus_downvotes_at_request']) 
    
    # We take logarithm of Karma, since the effect of Karma is more likely
    # to be logarithmic than linear.
    log_karma=[]
    for i in range(0, data_len):
        user_karma = karma[i]
        # A few users have negative Karma; that causes problems in
        # our taking of logs.
        if user_karma <= 0:
            user_karma = 0
        log_karma.append(math.log(user_karma+1))   
    
    # We concatenate the request title with the request text.
    reqtext = []
    reqtextlen = []
    for i in range(0, data_len):
        current_request = data[i]['request_title']+" "+\
        data[i]['request_text_edit_aware']
        reqtext.append(current_request.encode('ascii','replace'))
        reqtextlen.append(len(current_request)/req_length_factor)
        
    # We find the time of the request, measure since Dec 1 2014.
    # Earlier requests are more likely to have been satisfied.
    timestamp = []
    for i in range(0, data_len):     
        timestamp.append\
        ((data[i]['unix_timestamp_of_request_utc'] -\
        dec1_2010_timestamp)/timestamp_factor)
       
    # We define a dummy variable indicating whether or not the requester
    # posted in RAOP before making his request.
    raop_cred = []
    for i in range(0, data_len):
        if data[i]['requester_days_since_first_post_on_raop_at_request']>0:
            raop_cred.append(1)
        else:
            raop_cred.append(0)

    # Return the generated columns.    
    return [log_karma, reqtext, reqtextlen,timestamp, raop_cred]
                  
# Extract the dependent variable for the Train set.
def load_gotpizza(data):
    data_len = len(data)
    
    gotpizza=[]
    for i in range(0, data_len):
    	gotpizza.append(data[i]['requester_received_pizza'])
    
    # Turn the boolean variable to a dummy 0-1 variable.
    gotpizza10=[]
    for i in range(0, data_len):
    	if gotpizza[i]:
    		gotpizza10.append(1)
    	else:
    		gotpizza10.append(0)
    
    return gotpizza10

# Main method
def logit_train():
    
    # TRAIN    
    
    # LOAD TRAIN DATA    
    f = open('C:/games/Kaggle/pizza/train.json')
    data_train = json.load(f)
    f.close()

    data_train_len = len(data_train)    
    
    cols_train = load_columns(data_train)
    karma = cols_train[0]
    reqtext = cols_train[1]
    reqtextlen = cols_train[2]
    timestamp = cols_train[3]    
    raop_cred = cols_train[4]    
    
    gotpizza10 = load_gotpizza(data_train)    
    
    index=[]
    for i in range (0,data_train_len):
        index.append(data_train[i]['request_id'])    
    
    # GENERATE DICTIONARY OF KEYWORDS
    # Used to gather keywords.
    dictionary_pizza_1=dict()
    dictionary_pizza_0=dict()
    for i in range(0,data_train_len):
        cur_req = reqtext[i]
        extract_words = [w for w in re.split('\W', cur_req) if w]
        for word in extract_words:
            if gotpizza10[i]:
                if dictionary_pizza_1.has_key(word):
                    dictionary_pizza_1[word]+=1
                else:
                    dictionary_pizza_1[word]=1
            else:
                if dictionary_pizza_0.has_key(word):
                    dictionary_pizza_0[word]+=1
                else:
                    dictionary_pizza_0[word]=1
    
    # Sort dictionaries to find most frequently used words.
    sorteddict_1 = sorted(dictionary_pizza_1,\
        key=dictionary_pizza_1.get, reverse=True)
    sorteddict_0 = sorted(dictionary_pizza_0,\
        key=dictionary_pizza_0.get, reverse=True)
    
    # Remove useless words from our dictionaries.
    sifted_list_1 = [s for s in sorteddict_1 if not (s in useless_words)]
    sifted_list_0 = [s for s in sorteddict_0 if not (s in useless_words)]
    
    # Only retain the first retain_words_number words.
    sifted_list_1 = sifted_list_1[:retain_words_number]
    sifted_list_0 = sifted_list_0[:retain_words_number]    
    
    # Print sifted lists if desired.
    #print(sifted_list_1)
    #print(sifted_list_0)    
    
    # Many of the words in the Success and Failed lists overlap.
    # Here we count the number of overlaps, and generate lists
    # containing only the non-overlapping - and therefore significant - 
    # words.
    counter=0
    non_overlap_success = []
    non_overlap_fail = []
    for word in sifted_list_1:
        if word in sifted_list_0:
            counter+=1
        else:
            if not (word in success_words or word in sensitive_words):
                non_overlap_success.append(word)
    for word in sifted_list_0:
        if word in sifted_list_1:
            counter+=0
        else:
            if not word in fail_words:
                non_overlap_fail.append(word)
    
    # Print statements for the word lists.
    #print("OVERLAPPING WORDS COUNT:")
    #print(counter)
    #print("NON-OVERLAPPING SUCCESS WORDS:")
    #print(non_overlap_success)
    #print("NON-OVERLAPPING FAIL WORDS:")
    #print(non_overlap_fail)
        
    # OBTAIN KEYWORDS SCORE
    # We calculate scores for a given request on the basis of the keywords
    # present in the request text/title.
    sensitive_score=[]
    request_score_pos=[]
    request_score_neg=[]
    for i in range(0,data_train_len):
        cur_req = reqtext[i]
        extract_words = set([w for w in re.split('\W', cur_req) if w])
        score_pos=0    
        score_neg=0
        score_sensitive=0
        for word in extract_words:
            if word in success_words:
                score_pos+=1
            if word in fail_words:
                score_neg+=1
            if word in sensitive_words:
                score_sensitive+=1
                
        request_score_pos.append(score_pos)
        request_score_neg.append(score_neg)
        sensitive_score.append(score_sensitive)
    
    # We check if a request contains an image or a statement of gratitude.
    contains_img = []
    contains_thanks = []
    for i in range(0,data_train_len):
        cur_req = reqtext[i]
        extract_words = [w for w in re.split('\W', cur_req) if w]
        if "imgur" in extract_words:
            contains_img.append(1)
        else:
            contains_img.append(0)
            
        thanks = 0
        for thank_word in gratitude_words:
            if thank_word in extract_words:
                thanks = 1
        contains_thanks.append(thanks)

    # We check if the scores are heavily correlated with the request
    # lengths. The presense of such a heavy correlation is indicative 
    # of a`0 poor choice of keywords.    
    print("CORRELATION BETWEEN POS SCORE AND SENSE SCORE")
    print(np.corrcoef(request_score_pos, sensitive_score)[0, 1])    
    
    # PROCESS LOGIT REGRESSION
    
    # We generate the data frame for the logit.
    # In principle, we want to use a logit model, since the dependant variable
    # is binary. In practice, in this case the OLS model produces almost
    # equivalent results.
    ws = pd.DataFrame({
        'gotpizza': gotpizza10,
        'y1': karma,
        #'y2': reqtextlen, # Text length removed for having high p-value.
        'y3': request_score_pos,
        'y4': request_score_neg,
        'y5': sensitive_score,
        'y6': timestamp,
        'y7': raop_cred,
        'y8': contains_img,
        'y9': contains_thanks
    })
    ws['intercept'] = 1.0
    
    train_cols = ws.columns[1:]     
    logit = sm.Logit(ws['gotpizza'], ws[train_cols])
     
    # Fit the logit model and print the results.
    result = logit.fit()
    print(result.summary())
    
    # Also fit an OLS model and print its results.
    ols_fit = sm.ols('gotpizza ~ y1 +  y3 + y4 + y5 + y6 + y7 + y8 + y9',\
        data=ws).fit()
    print(ols_fit.summary())    
    
    # Generate the predicted probabilities of satisfying the requests.
    y_pred_logit = result.predict(ws[train_cols])
    y_pred_ols = ols_fit.predict(ws[train_cols])    
    
    # Dump the predicted probabilities so that we don't have to rerun the
    # algorithm during cutoff calibration.
    with open("C:/games/Kaggle/pizza/logit.dat", 'wb') as f:
        p.dump(y_pred_logit, f)
    
    # Save various pieces of data in a csv file for purposes of analysis.
    log_answers = [index, gotpizza10, y_pred_logit, y_pred_ols,karma,reqtextlen,\
    request_score_pos, request_score_neg, sensitive_score, timestamp,raop_cred,\
    contains_img,contains_thanks]
    with open('C:/games/Kaggle/pizza/train_check.csv', 'wb') as test_file:
        file_writer = csv.writer(test_file)
        file_writer.writerow(["request_id","requester_received_pizza",\
        "logit prob","ols prob","karma","reqtextlen","req score_post",\
        "req_score_neg", "score sensitive", "timestamp",\
        "raop_cred","img","thanks"])
        for i in range(data_train_len):
            file_writer.writerow([x[i] for x in log_answers])
    
    '''
    # LIST EXTRAORDINARY CASES:
    # Here we print bad cases of false positives and false negatives.
    for i in range(0,data_train_len):
        if gotpizza10[i]==0 and y_pred_logit[i]>0.7:
            print index[i]
    '''
    
    '''
    TEST
    '''    
    
    # Much of the following code duplicates code used
    # in the analysis of the Train set. It might be a good idea to
    # put common code in functions to avoid code duplication.
    
    # Load data and extract necessary columns.
    f = open('C:/games/Kaggle/pizza/test.json')
    data_test = json.load(f)
    f.close()
    
    data_test_len = len(data_test)    
    
    cols_test = load_columns(data_test)
    karma_test = cols_test[0]
    reqtext_test = cols_test[1]
    reqtextlen_test = cols_test[2]    
    timestamp_test = cols_test[3]
    raop_cred_test = cols_test[4]
    
    # Generate message scores for test dataset.
    request_score_pos_test=[]
    request_score_neg_test=[]
    sensitive_score_test=[]
    for i in range(0,data_test_len):
        cur_req = reqtext_test[i]
        extract_words = set([w for w in re.split('\W', cur_req) if w])
        cur_score_pos=0
        cur_score_neg=0
        cur_score_sensitive=0

        for word in extract_words:
            if word in success_words:
                cur_score_pos+=1
            if word in fail_words:
                cur_score_neg+=1
            if word in sensitive_words:
                cur_score_sensitive+=1
        
        request_score_pos_test.append(cur_score_pos)
        request_score_neg_test.append(cur_score_neg)
        sensitive_score_test.append(cur_score_sensitive)
    
    contains_img_test = []
    contains_thanks_test = []
    for i in range(0,data_test_len):
        cur_req = reqtext_test[i]
        extract_words = [w for w in re.split('\W', cur_req) if w]
        if "imgur" in extract_words:
            contains_img_test.append(1)
        else:
            contains_img_test.append(0)
    
        thanks = 0
        for thank_word in gratitude_words:
            if thank_word in extract_words:
                thanks = 1
        contains_thanks_test.append(thanks)
        
    Tws = pd.DataFrame({
        'y1': karma_test,
        #'y2': reqtextlen_test,
        'y3': request_score_pos_test,
        'y4': request_score_neg_test,
        'y5': sensitive_score_test,
        'y6': timestamp_test,
        'y7': raop_cred_test,
        'y8': contains_img_test,
        'y9': contains_thanks_test
    })
    Tws['intercept'] = 1.0    
    
    # Pick either logit or OLS prediction
    #Ty_pred = result.predict(Tws)
    Ty_pred = ols_fit.predict(Tws)
    #print(Ty_pred)  
    
    # Generate the prediction be checking against the cutoff coefficient.
    predict_test=[]    
    for i in range(0, data_test_len):
        q=0
        if Ty_pred[i] > cutoff_coeff:
            q=1
        predict_test.append(q)
        
    # GIVEAWAY TAKE ADVANTAGE
    # The 'giver_username_if_known' is a giveaway of about 100 ones
    # in the test set.
    counter=0
    for i in range(0, data_test_len):
        if data_test[i]['giver_username_if_known'] != "N/A":
            counter+=1
            predict_test[i]=1
            
    #print("GIVEAWAY BONUS")
    #print(counter)
     
    # Prepare prediction for writing to file.
    index_test=[]
    for i in range (0,data_test_len):
        index_test.append(data_test[i]['request_id']) 
        
    # Write prediction to file
    lol = [index_test, predict_test]
    with open('C:/games/Kaggle/pizza/test.csv', 'wb') as test_file:
        file_writer = csv.writer(test_file)
        file_writer.writerow(["request_id","requester_received_pizza"])
        for i in range(data_test_len):
            file_writer.writerow([x[i] for x in lol])
            
# Score Train prediction.
def verify_prediction(y_pred):
    f = open('C:/games/Kaggle/pizza/train.json')
    data = json.load(f)
    f.close()
    
    data_len = len(data)    
    
    gotpizza=[]
    for i in range(0, data_len):
    	gotpizza.append(data[i]['requester_received_pizza'])
    
    gotpizza10=[]
    for i in range(0, data_len):
    	if gotpizza[i]:
    		gotpizza10.append(1)
    	else:
    		gotpizza10.append(0)
    
    test1=[]   
    test2=[]
    for i in range(0, data_len):
        q=0
        if y_pred[i] > cutoff_coeff:
            q=1
        test1.append(q-gotpizza10[i])
        test2.append(q)
    
    score1 = sum(np.abs(test1))
    score2 = sum(test2)
    print("WRONG COUNT, OUT OF 4040")
    print(score1)    
    print("ONES COUNT, OUT OF 4040")
    print(score2)    
    
    return score1
  
# Comment/uncomment to switch running the algorithm off/on.  
logit_train()

# Score prediction. Use to calibrate cutoff value.
with open("C:/games/Kaggle/pizza/logit.dat", 'rb') as f:
    y_pred = p.load(f)

verify_prediction(y_pred)