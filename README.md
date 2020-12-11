# Predicting sleep state from wearable data

<b>Python 3.6</b> is used for this implementation beacuse of its easy way to work with data.<br />
Packages: <b>pandas</b>, <b>numpy</b>, <b>sklearn</b>, <b>xgboost</b>, <b>seaborn</b>

In order to run program sucessfully, firstly create a separated env, for example:<br />
<b>conda create -n sleep_classifier python=3.6</b> <br />
Then activate created environment:</br>
<b>conda activate sleep_classifier</b></br>
Then install all needed modules using:<br />
<b>pip install -r requirements.txt</b>

Running program from command line:<br />
CD yourself where the program script is, example below:<br />
<b>cd C:\Users\Danijel\Desktop</b>.<br />
and run the folowing command:<br />
<b>python sleep_classifier.py</b><br />

Additional comments can be found in the code.</br>

TO DO:
- implement n-fold cross-validation
- add additional metrics for evaluating performances of the models
