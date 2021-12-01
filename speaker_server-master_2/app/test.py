import math
import os
import random
import re
import sys

def gradingStudents(grades):
   arr = []
   for tmp in grades:
        if tmp >= 35:
            remain = tmp % 5
            if remain >=3:
                arr.append(tmp - remain + 5)
            else:
                arr.append(tmp)
        else:
            arr.append(tmp)
   return arrx
def breakingRecords(scores):
    # Write your code here
    best = scores[0]
    worst = scores[0]
    best_record = []
    worst_record = []
    for item in scores:
        if item > best:
            best = item
            best_record.append(1)
        else:
            best_record.append(0)
    for item in scores:
        if item < worst:
            worst = item
            worst_record.append(1)
        else:
            worst_record.append(0)
    result = []
    result.append(sum(best_record))
    result.append(sum(worst_record))
    return result
def birthday(s, d, m):
    # Write your code here
    count = 0
    if len(s) < m:
        return 0
    elif len(s) == m:
        if sum(s) == d:
            return 1
        else:
            return 0
    else:
        for i in range(0,len(s) - m + 1):
            total = 0
            for j in range(i,i+m):
                total = total + s[j]
            if total == d:
                count = count + 1
        return count
def sockMerchant(ar):
    # Write your code here
    dic = {}
    count = 0
    for tmp in ar:
        if tmp not in dic:
            dic[tmp] = 1
        else:
            dic[tmp] = dic[tmp] + 1
    for tmp in dic:
        count = count + dic[tmp] // 2
    print(count)
def countingValleys(path):
    # Write your code here
    step = 0
    valley = 0
    s = []
    s.append(0)
    for tmp in path:
        if tmp == "U":
            step = step + 1
        else:
            step = step - 1
        s.append(step)
    print(s)
    for i in range(0,len(s) - 1):
        if s[i]<0 and s[i+1] > 0:
            valley = valley + 1
    return valley
def migratoryBirds(arr):
    # Write your code here
    x = list(set(arr))
    y = sorted([(i,arr.count(i)) for i in x], key=lambda y: y[1],reverse=True)

    for i in range(len(y)):
        if y[0][0] == y[1][0] and y[0][1] > y[1][1]:
            return y[1][1]
        else:
            return y[0][1]

def bigSorting(unsorted):
    # Write your code here
    arr = []
    for item in unsorted:
        arr.append(int(item))
    a = sorted(arr)
    result = [str(i) for i in a]
    return result
def check(left,right):
    i = len(left)
    j = len(right)
    if i == j:
        if int(left) <= int(right):
            return True
        else:
            return False
    elif i <= j:
        return True
    else:
        return False

def partition(arr,low,high):
    i = low - 1
    pivot = arr[high]
    for j in range(low,high):
        if check(arr[j],pivot):
            i = i + 1
            arr[i],arr[j]=arr[j],arr[i]
    arr[i+1],arr[high] = arr[high],arr[i+1]
    return (i+1)



def quickSort(arr,low,high):
    if len(arr) == 1:
        return arr
    elif low < high:
        pi = partition(arr,low,high)
        quickSort(arr,low,pi-1)
        quickSort(arr,pi+1,high)
        return arr
def isValid(string):
    s = list(set(string))
    a = [string.count(i) for i in s]
    ss = list(set(a))
    aa = [a.count(i) for i in ss]
    if len(aa) == 1:
        return "YES"
    elif len(aa) == 2:
        if a.count(max(a)) == 1 and max(a) - min(a) == 1:
            return "YES"
        elif a.count(min(a)) == 1 and min(a) == 1:
            return "YES"
    return "NO"
def highestValuePalindrome(s,n,k):
    s = list(s)
    left = s[:n//2]
    right = s[n//2+1:][::-1] if n%2!=0 else s[n//2:][::-1]
    remainingCount = k
    diff = 0
    l = len(left)
    for i in range(l):
        if left[i] != right[i]:
            diff += 1
    for i in range(l):
        x = left[i]
        y = right[i]
        if x == y:
            if x == '9':
                continue
            if remainingCount - 2 >= diff:
                left[i] = right[i] = '9'
                remainingCount -= 1
        elif x != y:
           if x == '9':
               if remainingCount - 1 >= diff - 1:
                    right[i] = '9'
                    remainingCount -= 1
                    diff-=1
               else:
                   return -1
           elif y == '9':
               if remainingCount - 1 >= diff -1:
                   left[i] = '9'
                   remainingCount-=1
                   diff -= 1
               else:
                   return -1
           else:
               if remainingCount - 2 >= diff - 2:
                   left[i] = right[i] = '9'
                   remainingCount -= 2
                   diff-=2
               elif remainingCount - 1 >= diff - 1:
                   left[i] = right[i] = max(x,y)
                   remainingCount-=1
                   diff-=1
               else:
                   return -1
    if n%2 == 0:
        ans = "".join(left + (right[::-1]))
    else:
        mid = ['9'] if remainingCount>0 else [s[n//2]]
        ans = "".join(left + mid + (right[::-1]))
    return ans

















# !/bin/python3

import sys, math

MOD = 10 ** 9 + 7

fact = [1]
cur = 1
for i in range(1, 6 * 10 ** 4):
    cur *= i
    cur %= MOD
    fact.append(cur)

alpha = "abcdefghijklmnopqrstuvwxyz"

dp = [0]
dp2 = [[0] * 26]


def bitm(c):
    return 1 << (alpha.find(c))


def initialize(s):
    cur = 0
    for c in s:
        tmp = [i for i in dp2[-1]]
        tmp[alpha.find(c)] += 1
        cur ^= bitm(c)
        dp.append(cur)
        dp2.append(tmp)
    print(dp)


def modInv(x, pow=MOD - 2):
    if pow <= 3:
        return (x ** pow) % MOD
    else:
        tmp = modInv(x, pow // 2)
        tmp *= tmp
        tmp %= MOD
        if pow % 2 == 0:
            return tmp
        else:
            return (tmp * x) % MOD


mi = [modInv(fact[x]) for x in range(6 * 10 ** 4)]


def answerQuery(l, r):
    tmp = dp[l - 1] ^ dp[r]
    tmp2 = [(dp2[r][i] - dp2[l - 1][i]) // 2 for i in range(26)]
    possLetters = 0
    for c in range(26):
        if tmp & (1 << c):
            possLetters += 1
    middle = max(possLetters, 1)
    others = (r - l + 1 - possLetters) // 2
    ans = (fact[others] * middle) % MOD
    for i in tmp2:
        if i > 0:
            ans *= mi[i]
            ans %= MOD
    return ans
def removeAdjacent(s):
    for i in range(0, len(s) - 1):
        if s[i] == s[i + 1]:
            s.remove(s[i])
            s.remove(s[i + 1])

def superReducedString(s):
    s = list(s)
    if len(s) == 1:
        return s
    ind = 1
    while ind < len(s):
        if s[ind] == s[ind-1]:
            if(len(s)==2):
                return "Empty string"
            del s[ind-1:ind+1]
            ind = 1
        else:
            ind += 1
    if len(s) == 0:
        return "Empty string"
    else:
        return "".join(s)


def camelcase(s):
    # Write your code here
    s = list(s)
    count = 1
    for i in s:
        if i in ['A'-'Z']:
            count+=1
    print(count)
def minimumNumber( password):
    numbers = "0123456789"
    lower_case = "abcdefghijklmnopqrstuvwxyz"
    upper_case = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    special_characters = "!@#$%^&*()-+"
    password = list(password)
    arr = [0] * 4
    for i in password:
        if arr[0] != 1:
            if i in numbers:
                arr[0] = 1
                continue
        if arr[1] != 1:
            if i in lower_case:
                arr[1] = 1
                continue
        if arr[2] != 1:
            if i in upper_case:
                arr[2] = 1
                continue
        if arr[3] != 1:
            if i in special_characters:
                arr[3] = 1
                continue
    print(arr)
    if arr.count(0) >=  6 - len(s):
        count = arr.count(0)
    else:
        count = 6 - len(s)
    return count
def checkValid(cpy):
    for i in range(len(cpy) - 1):
        if cpy[i] == cpy[i + 1]:
            return False
    return True
def alternate(s):
    # Write your code here
    st = list(set(s))
    print(s)
    max_len = 0
    for i in range(len(st)):
        for j in range(i+1,len(st)):
            arr = [c for c in s if c == st[i] or c == st[j]]
            print(arr)
            if checkValid(arr):
                max_len = max(max_len,len(arr))
                print(max_len)
    return max_len

def twoSum(nums, target):
    dict = {}
    result = []
    for i in range(0,len(nums)):
        print(i)
        dict[nums[i]] = i
    for i in range(0,len(nums)):
        if(dict[target - nums[i]] != None and dict[target - nums[i]] != i):
            result.append(i)
            result.append(dict[target - nums[i]])
    return result




def shuffe(nums,n):
    arr = nums[n:]
    for i in range(0,2*n):
        if(i % 2 != 0):
            nums[i] = arr[(i-1)//2]
    return nums

def numIdenticalPairs(nums):
    x = {}
    print(x)
    enum
    for i in range(len(nums)):
        if (nums[i] in x):
            x[nums[i]].append(i)
        else:
            x[nums[i]] = [i]
    print(x)
    res = 0
    for i in x:
        n = len(x[i])
        if (n >= 2):
            y = math.factorial(n)
            z = math.factorial(n - 2)
            res += y // (z * 2)
    return res

def oddCells(m, n, indices):
    arr = [[0 for _ in range(0,n)] for _ in range(0,m)]
    for index in indices:
        for i in range(0, n):
            arr[index[0]][i] += 1
        for j in range(0,m):
            arr[j][index[1]]+=1
    print(arr)
def Power(n):
    if n == 0:
        return 1
    else:
        return Power(n-1) + Power(n-1)


def A(arr):
    if n==1:
        return arr[0]
    else:
        tmp = A(arr[:len(arr)-2])
        if tmp <= arr(len(arr)-1):
            return tmp
        else:
            return arr[len(n-1)]

def MaxIndex(myArray):
    if len(myArray) == 1: #Nếu myArray chỉ có 1 phần tử thì trả về vị trí đầu tiên của mảng
        return 0
    elif len(myArray) == 2: #Nếu myArray có 2 phần tử thì trả về vị trí của phần tử có giá trị lớn hơn
        if myArray[0] < myArray[1]:
            return 1
        else:
            return 0
    else:
        m = len(myArray) // 2 #Chia đôi độ dài của mảng
        aArray = myArray[0:m] #Lấy m phần tử đầu tiên của mảng
        bArray = myArray[m:len(myArray)] #Lấy m phần tử cuối của mảng
        largest_a = MaxIndex(aArray) #Gọi đệ quy cho m phần tử đầu tiên của mảng
        largest_b = MaxIndex(bArray) + m #Gọi đệ quy cho m phần tử cuối của mảng
        if myArray[largest_a] >= myArray[largest_b]: #Trả về chỉ số của phần tử có giá trị lớn hơn
            return largest_a
        else:
            return largest_b

def MinMax(A,MIN,MAX):
    if len(A) == 1: #Nếu mảng chỉ có 1 phần tử
        if MIN > A[0]:
            MIN = A[0]
        if MAX < A[0]:
            MAX = A[0]
        return MIN,MAX
    elif len(A) == 2: #Nếu mảng có 2 phần từ
        if A[0] < A[1]:
            if MIN > A[0]:
                MIN = A[0]
            if MAX < A[1]:
                MAX = A[1]
        else:
            if MIN < A[1]:
                MIN = A[1]
            if MAX > A[0]:
                MAX = A[0]
        return MIN, MAX
    else:
        mid = len(A) // 2 #Tìm phần tử ở giữ của mảng
        MIN, MAX = MinMax(A[0:mid],MIN,MAX) #Gọi đệ quy đối với nửa đầu của mảng
        MIN, MAX = MinMax(A[mid:len(A)-1],MIN,MAX) #Gọi đệ quy đối với nửa sau của mảng
        return MIN, MAX
def Power(a,n):
    if n == 1:
        return a
    elif n % 2 == 0:
        return Power(a,n//2) * Power(a,n//2)
    else:
        return a * Power(a, n // 2) * Power(a, n // 2)

import numpy as np
def sc_inversion_count(array):
    length = len(array)
    if length == 1:
        return array, 0
    if length == 2:
        if array[0] > array[1]:
            return np.array([array[1], array[0]]), 1
        else:
            return array, 0

    elif length > 2:
        array_l = array[:length // 2]
        array_r = array[length // 2:]
        array_l_df = sc_inversion_count(array_l)
        array_r_df = sc_inversion_count(array_r)
        array_l_sorted = array_l_df[0]
        array_r_sorted = array_r_df[0]
        length_l = len(array_l)
        length_r = len(array_r)
        count = array_l_df[1] + array_r_df[1]
        l = 0
        r = 0

        sorted_list = []

        for i in range(length):
            if r == length_r:
                sorted_list.append(array_l_sorted[l])
                l += 1
            elif l == length_l:
                sorted_list.append(array_r_sorted[r])
                r += 1

            elif array_l_sorted[l] > array_r_sorted[r]:
                sorted_list.append(array_r_sorted[r])
                r += 1
                count += len(array_l_sorted) - l

            elif array_l_sorted[l] < array_r_sorted[r]:
                sorted_list.append(array_l_sorted[l])
                l += 1

        return np.array(sorted_list), count

def countInvFirstType(L,R,lenL,lenR):
    ans = 0
    j = 0
    for i in range(0,lenL):
        while j < lenR and R[j] < L[i]:
            j = j + 1
        ans = ans + j
        j = 0
    return ans
def Merge(A,L,R):
    i = 0
    j = 0
    for id in range(0,len(L) + len(R)):
        if j >= len(R):
            A[id] = R[i]
            i = i + 1
        elif i >= len(L):
            A[id] = L[j]
            j = j + 1
        elif L[i] <= R[j]:
            A[id] = L[i]
            i = i +1
        else:
            A[id] = R[j]
            j = j + 1
    return A
def countInversion(A,n):
    ans = 0
    if n <= 1:
        return 0
    lenL = n // 2
    lenR = lenL + 1
    L = A[:lenL]
    R = A[lenR:]
    ans += countInversion(L,lenL)
    ans += countInversion(R,lenR)
    ans += countInvFirstType(L,R,lenL,lenR)
    Merge(A,L,R)
    return ans


def mergeSort(A):
    if len(A) == 1:                           #Nếu số phần tử trong A = 1 thì trả vể mảng A và inversion = 0
        return A , 0
    mid = len(A) // 2                         #Chia đôi mảng A
    L = A[:mid]
    R = A[mid:]
    firstHalf, inversion1 = mergeSort(L)      #Gọi đệ quy cho nửa đầu của mảng A
    secondHalf, inversion2 = mergeSort(R)     #Gọi đệ quy cho nửa sau của mảng A

    sortedArray = []
    i = 0
    j = 0
    inversionCount = 0
    while i < len(firstHalf) and j < len(secondHalf):   #So sánh từng phần tử trong firstHalf và secondHalf
        if firstHalf[i] > secondHalf[j]:                #Nếu firstHalf[i] > secondHalf[j] thì là inversion
            sortedArray.append(secondHalf[j])           #Thêm secondHalf[j] vào mảng sortedArray
            inversionCount += len(firstHalf) - i        #inversion bằng số các phần tử nằm phía bên phải của firstHalf
            j += 1                                      #Tăng biến đếm j lên 1
        else:
            sortedArray.append(firstHalf[i])            #Nếu secondHalf[j] > firstHalf[i] thì là không phải là inversion
            i += 1
    while i < len(firstHalf):
        sortedArray.append(firstHalf[i])                #Thêm tất cả các phần tử còn lại trong firstHalf vào sortedArray
        i+=1
    while j < len(secondHalf):
        sortedArray.append(secondHalf[j])               #Thêm tất cả các phần tử còn lại trong secondHalf vào sortedArray
        j+=1
    totalInversion = inversion1 + inversion2 + inversionCount  #tổng các inversion
    return sortedArray, totalInversion

def bruce_force(A):
    inversions = 0
    for i in range(0,len(A)):
        for j in range(i+1, len(A)):
            if A[i] > A[j]:
                inversions += 1
    return inversions
def merge(A,temp,fro,mid,to):
    k = fro
    i = fro
    j = mid + 1
    while i <= mid and j <= to:
        if A[i] < A[j]:
            temp[k] = A[i]
            i += 1
        else:
            temp[k] = A[j]
            j += 1
        k += 1
    while i < len(A) and i <= mid:
        temp[k] = A[i]
        k = k + 1
        i = i + 1
    for i in range(fro, to + 1):
        A[i] = temp[i]
def Bottom_up_Merge_Sort(A):
    low = 0                      #Lấy chỉ số phần tử đầu tiên của mảng A
    high = len(A) - 1            #Lấy chỉ số của phần tử cuối cùng mảng A
    temp = A.copy()              #khai báo 1 mảng tạm thời tmp
    m = 1
    while m <= high - low:
        for i in range(low,high,2*m):     #Chia mảng A thành từng khối có kích thước từ size = [1,2,4,8,16...]
            fro = i
            mid = i + m - 1
            to = min(i+2*m-1,high)
            merge(A,temp,fro,mid,to)      #sắp xếp các phần tử trong mỗi khối
        m=m*2                             #Tăng kích thước của khối lên 2 lần
    return A
def partitionL(A,low,high): # using Lomuto's partition
    pivot = A[high]
    i = low - 1
    for j in range(low,high):
        if A[j] <= pivot:
            i = i + 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[high] = A[high], A[i + 1]
    return (i+1)


def swap(A, i, j):
    temp = A[i]
    A[i] = A[j]
    A[j] = temp



def partitionH(a, low, high): # Partition using Hoare's Partitioning scheme
    pivot = a[low]
    (i, j) = (low - 1, high + 1)
    while True:
        while True:
            i = i + 1
            if a[i] >= pivot:
                break
        while True:
            j = j - 1
            if a[j] <= pivot:
                break
        if i >= j:
            return j
        swap(a, i, j)
def quickSort(A,low,high):
    if low < high:
        pivot = partitionH(A,low, high)
        quickSort(A,low,pivot-1)
        quickSort(A,pivot+1,high)
def RobotCoinCollecting(C):
    infinite = -10000000000
    print(C[0][0])
    F = A.copy()      #Khỏi tạo 1 ma trận mới có kích thước bằng ma trận đầu vào
    F[0][0] = C[0][0] #Gán giá trị tại F[0,0] = C[0,0]
    for j in range(1,len(C[0])): #Duyệt các phần tử đầu tiên của hàng thứ nhất
        if C[0][j] == -1:
            F[0][j] = infinite   #Nếu ô nào không thể truy cập được thì gán giá trị cho nó bằng âm vô cùng
        else:
            F[0][j] = F[0][j-1] + C[0][j]   #Nếu ô nào có thể truy cập được thì giá trị của nó bằng tổng giá trị của ô liền trước nó với giá trị của hiện tại
    for i in range(1,len(C)):    #duyệt các hàng còn lại của ma trận
        if C[i][0] == -1:
            F[i][0] = infinite   #Nếu ô nào không thể truy cập được thì gán giá trị cho nó bằng âm vô cùng
        else:
            F[i][0] = F[i-1][0] + C[i][0]
        for j in range(1,len(C[0])):
            if F[i-1][j] == infinite and F[i][j-1] == infinite: #Nếu cả ô liền trước bên trái và ô phía trên ô hiện tại đều không thể truy cập thì trả về None
                return None
            else:
                F[i][j] = max(F[i-1][j],F[i][j-1]) + C[i][j]    #Tìm giá trị lớn nhất trong 2 ô liền trước bên trái với ô phía trên rồi cộng với giá trị của ô hiện tại
    return F   #Trả về F nếu có đường đi tới ô dưới cùng bên phải
def uniquePaths(A):
    # Tạo mảng 2 chiều và khởi tạo giá trị bằng 0
    paths = [[0 for i in range(len(A[0]))] for j in range(len(A))]
    if A[0][0] == 0: #khởi tạo giá trị cho ô góc trên phía bên trái nếu không gặp chướng ngại vật
        paths[0][0] = 1
    if i in range(1,len(A)):#Khởi tạo giá trị cho cột đầu tiên
        if A[i][0] == 0: #Nếu không gặp chướng ngại vật
            paths[i][0] = paths[i-1][0]
    for j in range(1,len(A[0])):#Khởi tạo giá trị cho hàng đầu tiên
        if A[0][j] == 0: #Nếu không gặp chướng ngại vật
            paths[0][j] = paths[0][j-1]
    for i in range(1, len(A)):
        for j in range(1,len(A[0])):
            if A[i][j] == 0: #Nếu không gặp chướng ngại vật
                paths[i][j] == paths[i-1][j] + paths[i][j-1]
    return paths[-1][-1]
import sys
MAX_VAL = -sys.maxsize - 1
def cutRod(price, n):
    val = [0 for x in range(n+1)] # khởi tạo mảng có kích thước n lưu giá trị lớn nhất của các thanh
    val[0] = 0                    # giá trị thanh có chiều dài 0 là bằng 0
    for i in range(1,n+1):        # Duyệt i từ thanh có độ dài 1 đến n
        max_val = MAX_VAL
        for j in range(i):        # Gía trị lớn nhất của thanh khi có độ dài bằng i
            max_val = max(max_val, price[j]+val[i-j-1])
        val[i]=max_val            # gán giá trị lớn nhất đó vào độ dài tương ứng của thanh trong mảng val
    return val[n]

def minSumPath(A):
    levels = len(A)     
    array = [[0 for i in range(len(A))] for j in range(len(A))]
    array[levels - 1] = A[levels-1]
    for i in range(levels-2,-1,-1):
        for j in range(0,i+1):
            array[i][j] = A[i][j] + min(array[i+1][j],array[i+1][j+1])
    return array
def binomialCoeff(n,k):
    C = [[0 for x in range(k+1)] for y in range(n+1)] #Khai báo ma trận 2 chiều có kích thước (nxk) để lưu kết quả
    for i in range(n+1):     #Tính giá trị của hệ số Binomial
        for j in range(min(i,k)+1):
            if j==0 or j==i: #trong trường hợp j=0 hoặc j=i thì giá trị của hệ số Binomial = 1
                C[i][j] = 1
            else:
                C[i][j] = C[i-1][j-1] + C[i-1][j] # Trong trường hợp còn lại thì hệ sô Binomial tạo (i,j) = (i-1,j-1) + (i-1,j)
    return C[n][k]  #hệ số C(k,n) bằng giá trị tại hàng n cột k
def MaxSubSquare(M):
    S = [[0 for i in range(len(M)+1)] for j in range(len(M[0])+1)] #khởi tạo ma trân phụ có kích thước bằng với kích thước của ma trận ban đầu để lưu kết quả
    largest = 0 #Khởi tạo giá trị largest ban đầu bằng 0
    for i in range(1,len(M)+1): #Duyệt tất cả các ô không thuộc hàng 1 và cột 1
        for j in range(1,len(M[0])+1):
            if M[i][j] == 0: #Nếu ô hiện tại bằng 0 thì set giá trị cho S[i][j]
                S[i][j] = 1 + min(S[i-1][j],S[i][j-1],S[i][j])
                if S[i][j] > largest: #Nếu S[i][j] lớn hơn largest thì largest == S[i][j]
                    largest = S[i][j]
            else: #Ngược lại thì set S[i][j] = 0
                S[i][j] = 0
    return largest
def coin_Row(C):
    F = [0 for i in range(len(C)+1)] #Khởi tạo mảng F để lưu các kết quả
    F[0] = 0      #giá trị khởi tạo F[0] = 0
    F[1] = C[0]   #giá trị khỏi tạo F[1] = C[0]
    for i in range(2,len(C)+1): # Duyệt từ phần tử thứ 3 đến cuối mảng
        F[i] = max(C[i-1]+F[i-2],F[i-1]) #Cập nhật kết quả tính được vào mảng F
    return F[len(C)] # Trả về giá trị cuối cùng của bài toán
def changMaking(D,n):
    F = [0 for i in range(n+1)] #Khởi tạo mảng F để lưu kết quả
    F[0] = 0  # giá trị khỏi tạo F[0] = 0
    for i in range(1,n+1): #Duyệt từ 1 đến hết mảng
        temp = 1000000
        j = 1
        #Duyệt các mệnh giá có trong mảng D thõa mãn điều kiện j nhỏ hơn chiều dài mảng
        #và giá trị hiện tại của i phải lớn hơn mệnh giá tại vị trí thứ j
        while j <= len(D)-1 and i >= D[j]:
            temp = min(F[i-D[j]],temp)
            j = j + 1
        F[i] = temp + 1
    return F[n]








#
# if __name__ == '__main__':
#     # A = [[0,-1,0,1,0,0],[-1,0,0,-1,1,0],[0,1,0,-1,1,0],[0,0,0,1,0,1],[-1,-1,-1,0,1,0]]
#     # print(RobotCoinCollecting(A))
#     # arr = [1, 5, 8, 9, 10, 17, 17, 20]
#     # size = len(arr)
#     # print(cutRod(arr,size))
#     # A = [[2],
#     #      [3, 9],
#     #      [1, 6, 7]]
#     # print(minSumPath(A))
#     print(changMaking([1,3,4],6))


def swap(A, i, j):
    temp = A[i]
    A[i] = A[j]
    A[j] = temp


# Partition using Hoare's Partitioning scheme
def partition(a, low, high):
    pivot = a[low]
    (i, j) = (low - 1, high + 1)

    while True:

        while True:
            i = i + 1
            if a[i] >= pivot:
                break

        while True:
            j = j - 1
            if a[j] <= pivot:
                break

        if i >= j:
            return j

        swap(a, i, j)


# Quicksort routine
def quicksort(a, low, high):
    # base condition
    if low >= high:
        return

    # rearrange elements across pivot
    pivot = partition(a, low, high)

    # recur on sublist containing elements less than the pivot
    quicksort(a, low, pivot)

    # recur on sublist containing elements more than the pivot
    quicksort(a, pivot + 1, high)


if __name__ == '__main__':
    a = [5, 3, 1, 9, 8, 2, 4, 7]

    quicksort(a, 0, len(a) - 1)

    # print the sorted list
    print(a)







