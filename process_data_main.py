from scripts.process_temp import process_temp_data
from scripts.utils import get_santander_boundaries

def main():
    data_temp = process_temp_data()
    data_temp.to_excel("data/tmean.xlsx", index=False)
    get_santander_boundaries()

if __name__ == "__main__":
    main()
    
