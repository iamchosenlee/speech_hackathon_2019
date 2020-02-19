import re

##################################### 알파벳
ALPH = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','&']
ALPH2KOR = ['에이','비','씨','디','이','에프','지','에이치','아이','제이','케이',
        '엘','엠','엔','오','피','큐','알','에스','티','유','브이','더블유','엑스','와이','제트','엔']

def convAlphabet(match):
    kor = ''
    for alpha in match:
        if alpha.lower() in ALPH:
            kor += ALPH2KOR[ALPH.index(alpha)]
    return kor

def read_alpha(string):
    try:
        alpha = re.compile('[A-Za-z]{1}')
        alp = alpha.findall(string)
        #print(alp)
        result= string.replace(''.join(alp),convAlphabet(alp))
    except:
        return string
    return result


#print(read_alpha(t5))




##################################전화번호
readNumber= ['공','일','이','삼','사','오','육','칠','팔','구']
def number2readNumber(num_phone):
    ph = ''
    num_phone = ''.join(num_phone)
    for number in num_phone:
        #print(number)
        ph += readNumber[int(number)]
    return ph

phone = re.compile(r'(\d\d\d)-(\d\d\d\d)-(\d\d\d\d)')
def read_phone(string):
    try:
        p_phone = re.compile(r'(\d\d\d)-(\d\d\d\d)-(\d\d\d\d)')
        phone_string = re.compile(r'\d\d\d-\d\d\d\d-\d\d\d\d').findall(string)
        phone_number = list(p_phone.findall(string)[0])
        #print(phone_number)
        result = string.replace(phone_string[0], number2readNumber(phone_number))
    except:
        return string
    return result

#print(read_phone(t3))


########################### 기수


gisu = re.compile('([0-9]+)(?:시|명|잔|개)')
# gi_num = gisu.findall(t1)

#print(gi_num)
def num2gisu(n):
    units = [''] + list('열')
    #nums = '일이삼사오육칠팔구'
    readCount = ['', '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉']
    result = ''
    i=0

    if n==20 :
        result = '스무'
    elif n<20 :
        n, r = divmod(n,10)
        if n>0 :
            result = '열' + readCount[r]
        else :
            result = readCount[r]
    else :
        result = str(n)
    return result

def read_gisu(string):
    try:
        gisu = re.compile('([0-9]+)(?:시|명|잔|개)')
        gi_num = gisu.findall(string)
        for i in gi_num:
            #print(seo_num)
            string = string.replace(''.join(i), num2gisu(int(i)))
    except:
        return string
    return string


#print("gisu", read_gisu(t1))

##################### 서수
seosu = re.compile('([0-9]+)(?:분|만원|동|호|인|g|월|일|층)')
#seo_num = seosu.findall(t4)
#print(seo_num)


def num2seosu(n):
    units = [''] + list('십백천')
    nums = '일이삼사오육칠팔구'
    result = []

    i=0
    while n>0:
        n, r = divmod(n,10)
        if r >0:
            result.append(nums[r-1] + units[i])
        i +=1
    #print(result)
    result = list(''.join(result[::-1]))
    #print(result)
    if (len(result) >1) and (result[0] == '일'):
        del result[0]

    return ''.join(result)

print(num2seosu(129))

def read_seosu(string):
    try:
        seosu = re.compile('([0-9]+)(?:분|만원|동|호|인|g|월)')
        seo_num = seosu.findall(string)
        for i in seo_num:
            #print(seo_num)
            string = string.replace(''.join(i), num2seosu(int(i)))
    except:
        return string
    return string

#print(read_seosu(t4))



###################
bunji = re.compile(r'(\d\d\d-\d\d)')
def read_bunji(string):
    try:
        bun_num = re.compile(r'(\d\d\d)-(\d\d)')
        bun_str = re.compile(r'(\d\d\d-\d\d)').findall(string)
        bunji_list = list(bun_num.findall(string)[0])
        bunji_str = num2seosu(int(bunji_list[0]))+'다시'+num2seosu(int(bunji_list[1]))
        result = string.replace(bun_str[0], bunji_str)
    except:
        return string
    return result

#print(read_bunji('310-10번지입니다'))