import data_io
import pickle

def my_tokenizer(s):
    return s.split('?')


def main():
    
    valid = data_io.get_valid_df()
    P={}
    for key in valid:
        print("Loading the classifier for %s" %key)
        classifier = data_io.load_model(key)  
        print("Making predictions") 
        P[key] = classifier.predict(valid[key])   
        P[key] = P[key].reshape(len(P[key]), 1)

    print("Writing predictions to file")
    data_io.write_submission(P)

if __name__=="__main__":
    main()