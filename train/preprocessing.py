import utils.preprocess as pp
import utils.settings as st

def main():
    pp.txt_to_csv(
        st.RAW_DATA_PATH, 
        st.RAW_DATA_PROCESSED_PATH
    )
    pp.get_train_and_validation_data(
        st.RAW_DATA_PROCESSED_PATH, 
        st.TRAIN_DATA_PATH, 
        st.VALIDATION_DATA_PATH, 
        validation_size=0.1, 
        save_files = False
    )
if __name__ == "__main__":
    main()