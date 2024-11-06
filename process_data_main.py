from scripts.process_temp import process_temp_data

def main():
    data_temp = process_temp_data()
    data_temp.to_excel("data/tmean.xlsx", index=False)

if __name__ == "__main__":
    main()
    
