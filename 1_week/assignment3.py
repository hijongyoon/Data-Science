def main():
    f = open("cars.csv", "r")
    f.readline()
    car_list = make_list_of_car(f)
    print("1.")
    print_car_info(car_list)
    print("2.")
    selected_car = [x for x in car_list if "20000" <= x["Price"] <= "50000" and
                    int(x["Year"]) >= 2000 and x["Body"] == "sedan" and x["Engine_Type"] == "Gas"]
    selected_car.sort(key=lambda dic: int(dic["Price"]))
    print_car_info(selected_car)
    print("3.")
    car_name_set = set()
    for i in car_list: car_name_set.add(i["Brand"])
    print_car_info(car_name_set)


def make_list_of_car(f):
    car_list_of_dict = []
    for car in f.readlines():
        car = car.split(",")
        car_info = {"Brand": car[0], "Price": car[1], "Body": car[2], "Mileage": car[3], "EngineV": car[4],
                    "Engine_Type": car[5], "Registration": car[6], "Year": car[7], "Model": car[8].rstrip()}
        car_list_of_dict.append(car_info)
    return car_list_of_dict


def print_car_info(li):
    for i in li:
        print(i)


main()
