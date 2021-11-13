'''
Main code for the UOC Hackaton

Author:
    Lucas Goiriz Beltran
'''

# Global imports
import argparse, sys
from io import StringIO
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

################################################################################
#                                                                              #
#                          Definition of funnctions                            #
#                                                                              #
################################################################################

def main(df1, df2):
    '''Create Random Forest from given data'''
    
    # Select the predictors. Basically feature1, feature2, ... , feature8
    predictors = df1[df1.columns[df1.columns!= "target"]]

    # Select the labels
    labels = df1[["target"]].T.to_numpy()[0]
    
    # Train the model
    model = RandomForestClassifier(n_estimators=200)
    model.fit(predictors, labels)
    
    # Predict
    pred = model.predict(df2)
    
    return pd.DataFrame(data=pred, columns=["predicitons"])


def argparser():
    '''Argparser for easy code execution'''

    parser = argparse.ArgumentParser(
        description="Create and run a Random Forest on some data."
    )
    parser.add_argument(
        "train_file", metavar="trnf", type=str,
        help="Filepath of dataset used to train the model. Must be in `.csv`."
    )
    parser.add_argument(
        "test_file", metavar="tstf", type=str,
        help="Filepath of dataset used to test the model. Must be in `.csv`."
    )

    # Check if script was executed without inputs
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()
    
    # Check if inputs are csv files
    if (
        (not args.train_file.endswith('.csv'))
        or (not args.train_file.endswith('.csv'))
    ):
        raise parser.error("Input files must be `.csv`.")
        
    # Check whether these files are parseable
    try:
        with open(args.train_file) as file:
            train_file_contents = ''.join(file.readlines())
    except FileNotFoundError:
        raise parser.error("No such file: `{}`".format(args.train_file))
        
    try:
        with open(args.test_file) as file:
            test_file_contents = ''.join(file.readlines())
    except FileNotFoundError:
        raise parser.error("No such file: `{}`".format(args.test_file))
    
    return (train_file_contents, test_file_contents)



################################################################################
#                                                                              #
#                             Standalone execute                               #
#                                                                              #
################################################################################

if __name__ == "__main__":
    
    # Parse inputs
    (train_file, test_file) = argparser()

    df1 = pd.read_csv(StringIO(train_file))
    df2 = pd.read_csv(StringIO(test_file))
    
    del train_file, test_file

    # Execute model and fit
    df3 = main(df1, df2)

    # Export file
    df3.to_csv("fit_output.csv", index=False)
    
    print("Output `fit_output.csv` created.\nByebye!")
