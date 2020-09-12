def main():
    f = open("cars.csv", "r")
    f.readline()
    data = f.readlines()
    li1 = [x for x in data if x.split(",")[0] == "Volkswagen" and x.split(",")[5] == "Gas" and x.split(",")[2] == "sedan"]
    print(len(li1))
    li2 = [x.split(",") for x in data if x.split(",")[0] == "BMW"]
    li2.sort(key=lambda li: int(li[7]))
    print(li2)
    f.close()


main()
