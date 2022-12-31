from src import cyclegan_with_monitoring_generic

def main():
    try:
        #Add dataset to
        cyclegan_with_monitoring_generic()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()

