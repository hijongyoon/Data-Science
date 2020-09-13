li = [3, 6, 8, 10, 1, 2, 1]


def insert_sorting(li):
    for index in range(1, len(li)):
        while index - 1 >= 0 and li[index] < li[index - 1]:
            li[index], li[index - 1] = li[index - 1], li[index]
            index -= 1
    return li


def merge_sorting(start, end):
    global li
    if start < end:
        merge_sorting(start, (start + end) // 2)
        merge_sorting((start + end) // 2 + 1, end)
        merge(start, (start + end) // 2, end)
    return li


def merge(start, middle, end):
    global li
    copy = [0 for x in range(len(li))]
    i, j, k = start, middle + 1, start
    while i <= middle and j <= end:
        if li[i] <= li[j]:
            copy[k] = li[i]
            k += 1
            i += 1
        else:
            copy[k] = li[j]
            k += 1
            j += 1
    while i <= middle:
        copy[k] = li[i]
        k += 1
        i += 1
    while j <= end:
        copy[k] = li[j]
        k += 1
        j += 1
    for p in range(start, end + 1):
        li[p] = copy[p]


def main():
    print(merge_sorting(0, len(li) - 1))
    print(insert_sorting([3, 6, 8, 10, 1, 2, 1]))


main()
