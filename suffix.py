import re
import math
import string
import time
text = open("TALES.txt",'r')
text1 = open("TALES.txt",'r')

k=[]
tlist = []
dic = {}
blank=0
story = ''
titles = []
Text = text.read()
#print(titles)
for i in text1:
	exclude = set(string.punctuation)-{'?','-'}
	i = ''.join(ch for ch in i if ch not in exclude)
	if(blank == 2):
		blank=0
		s = '\n'.join(k)
		#print(s)
		dic[str(story)] ='\n'+s
		k =[]		 
		story = i[:-2]
		#print(story)	
	if (len(i.strip())!=0 and i[:-2] != story):
		blank = 0
		k.append(i[:-1])
	if(len(i.strip())==0):
		blank = blank+1
	else:
		blank = 0
dic[story] = '\n'+'\n'.join(k)
del dic['']
#print(dic['FOOTNOTES'].split())
pT = 0
def Rabin_Karp_Matcher(text,pattern,start,end,d, q):
    global pT
    n = (end-start)+1
    m = len(pattern)
    h = pow(d,m-1)%q
    p = 0
    t = 0
    result = []
    starttime = time.time()
    for i in range(m): 
        p = (d*p+ord(pattern[i]))%q
    for i in range(m): 
        t = (d*t+ord(text[i+start]))%q
    endtime = time.time()
    pT = endtime - starttime
    for s in range(n-m+1): 
        if p == t: 
            match = True
            for i in range(m):
                if pattern[i] != text[s+i+start]:
                    match = False
                    break
            if match:
                result = result + [s]
        if s < n-m:
            """t = (t-h*ord(text[s]))%q 
            t = (t*d+ord(text[s+m]))%q 
            t = (t+q)%q"""
            t = ((d*(t-ord(text[s+start])*h))+ord(text[s+m+start]))%q 
		
    return result
	
def kmpMatcher(t,p,start,end):
	global pT
	n = (end-start)+1
	m = len(p)
	starttime = time.time()
	pi = computePrefix(p)
	endtime = time.time()
	pT = endtime - starttime
	q = -1
	result = []
	for i in range(n):
		while(q>-1 and p[q+1] != t[i+start]):
			q = pi[q]
		#print("hello")
		if(p[q+1] == t[i+start]):
			q = q+1
		if(q==m-1):
			result = result + [i-(m-1)]			
			q = pi[q]
	return result


def computePrefix(p):
	m = len(p)
	pi=[0 for i in range(m)]
	pi[0] = -1
	k = -1
	for q in range(1,m):
		while k>-1 and p[k+1] != p[q]:
			k = pi[k]
		if(p[k+1] == p[q]):
			k = k+1
		pi[q] = k
	return pi


def build_suffix_array(text):
	text_length = len(text)
	suffixes = []
	for i in range(text_length):
		suffixes.append([])
		suffixes[i].append(i)
		suffixes[i].append(ord(text[i]) - ord('a'))
		if i + 1 < text_length:
			suffixes[i].append(ord(text[i+1]) - ord('a'))
		else:
			suffixes[i].append(-1)
	suffixes = sorted(suffixes , key= lambda x : (x[1],x[2]))		
	ind = []
	for i in range(text_length):
		ind.append(0)
	k = 4
	while(k < 2 * text_length):
		rank = 0
		p_rank = suffixes[0][1]
		suffixes[0][1] = rank
		ind[suffixes[0][0]] = 0
		for i in range(1,text_length):
			if(suffixes[i][1] == p_rank and suffixes[i][2] == suffixes[i-1][2]):
				p_rank = suffixes[i][1]
				suffixes[i][1] = rank
			else:
				p_rank = suffixes[i][1]
				rank += 1
				suffixes[i][1] = rank
			ind[suffixes[i][0]] = i
		for i in range(text_length):
			n_i = int(suffixes[i][0] + k/2);
			if(n_i < text_length):
				suffixes[i][2]=suffixes[ind[n_i]][1]
			else:
				suffixes[i][2] = -1	 		
		suffixes = sorted(suffixes , key= lambda x : (x[1],x[2]))		
		k = k * 2
	suffix_array = []		
	for i in range(text_length):
		suffix_array.append(suffixes[i][0])
	return suffix_array

def lcp(string_one , string_two):
	n = len(string_two)
	number = 0
	i = 0 
	while(  i < n and string_one[i] == string_two[i]):
		i+=1	
	return i	

def suffix_pattern(pattern ,t):
	count = 0 
	s_a=build_suffix_array(t)
	#print(s_a)
	L = 0 #indicates the starting index of the suffix array..
	R = len(s_a) - 1 # indicates the last index of the suffix array..
	L1 = L
	R1 = R
	lower_bound = 0 
	p_len = len(pattern)
	#print(p_len)
	if( pattern < t[s_a[L]: s_a[L] + p_len]):
		lower_bound = 0
	elif( pattern > t[s_a[R]:s_a[R] + p_len ]):
		lower_bound = R + 1
	else:
		while(R - L >= 0):
			M = math.ceil((R + L) / 2)
			M = int(M)
			#print(M)
			if(pattern <= t[s_a[M] : s_a[M]+p_len]):
				R = M - 1
			else:
				L = M + 1
		lower_bound = R + 1
	done = True
	while(lower_bound < R1 + 1 and done):
		if(lcp(pattern , t[s_a[lower_bound] : s_a[lower_bound] + p_len ]) == p_len):
			lower_bound += 1
			count += 1
		else:
			done = False
	return count		

def find_pattern():
	
	k = int(input("1 -> for entering starting and ending indices \n2 -> for entering story titles\n"))
	if(k==1):   
		start = int(input("Enter start index:"))
		end = int(input("Enter end index:"))
	else:
		start = input("Enter 1st title:")
		end = input("Enter 2nd title:")
		start  = Text.index(start)
		end = Text.index(end)
		if(start > end):
			start,end = Text.index(end),Text.index(start)

	pattern = input("Enter the pattern:")
	print("Pick Algorithm to search for pattern")
	print("1->Rabin karp\n2->KMP algorithm\n3- Suffix")
	choice = int(input())
	if(choice == 1):
		startTime = time.time()
		print("Number of occurences by Rabin Karp:",len(Rabin_Karp_Matcher(Text,pattern,start,end,26,97)))
		endTime = time.time()
		return (endTime - startTime),"Rabin Karp",pattern
	elif(choice == 2):
		startTime = time.time()
		print("Number of occurences by KMP:",len(kmpMatcher(Text,pattern,start,end)))
		endTime = time.time()
		return (endTime - startTime),"KMP algorithm",pattern

	elif(choice == 3):
		inp1 = Text[start:end]
		startTime = time.time()
		print("The number of occurences of the pattern",pattern," is : " , suffix_pattern(pattern ,inp1))
		endTime = time.time()
		return (endTime - startTime),"Suffix Array",pattern

 
		
def palindrome(string,maxlen):
	palindromeList = []
	length = len(string)
	suffDict = {}
	lcp = []
	suffArray = []
	suffLexic = []
	rev = string[::-1]
	string = string+"$"+rev
	#print(string)
	suffDict[string] = 1
	suffLexic.append(string) 
	for i in range(1,len(string)):
		suffDict[string[i:]] = i+1 
		suffLexic.append(string[i:])
	#print(suffDict)
	suffLexic = sorted(suffLexic, key=str.lower)
	for i in suffLexic:
		suffArray.append(suffDict[i])
	#print(suffArray)
	#print(suffLexic)
	lcp.append(0)
	i=1
	while i<(len(suffLexic)):
		k = 0
		count = 0
		while (k < len(suffLexic[i]) and k < len(suffLexic[i-1]) and suffLexic[i][k] == suffLexic[i-1][k]) :
			count = count+1
			k = k+1
		#print(count)
		lcp.append(count)
		i = i+1
	#print("LCP",lcp)
			
	i=0
	maxv = 0
	mindex = []
	found = False
	index = 0
	iarray = []
	LCP = lcp
	while(found== False):
		i = 0
		maxv = 0
		index = 0
		while i < len(lcp):
			if(lcp[i]>maxv and (i not in mindex)):
				maxv = lcp[i]
				index = i
			i = i+1
		iarray = [i for i,x in enumerate(lcp) if x==maxv]
		for i in iarray:
			mindex.append(i)
		for index in iarray:
				if((string.endswith(suffLexic[index-1]) and (len(suffLexic[index-1]) > length)  and string.endswith(suffLexic[index],length+1)) or (string.endswith(suffLexic[index]) and (len(suffLexic[index]) > length) and string.endswith(suffLexic[index-1],length+1)) ):
				
					found = True
					if(maxv>=maxlen):
						palindromeList.append(suffLexic[index-1][:maxv])


		palindromeFinal = []
		for i in palindromeList :
			k = 1
			palindromeFinal.append(i)
			while k <= (len(i)//2):
				s = i[k:(len(i)-k)]
				if(len(s)>=maxlen):
					palindromeFinal.append(s)
				k = k+1
			
	return palindromeFinal

def findPalindrome():
	
	k = int(input("1 -> for entering starting and ending indices \n2 -> for entering story titles\n"))
	if(k==1):   
		start = int(input("Enter start index:"))
		end = int(input("Enter end index:"))
	else:
		start = input("Enter 1st title:")
		end = input("Enter 2nd title:")
		start = Text.index(start)
		end = Text.index(end)
		if(start > end):
			start,end = Text.index(end),Text.index(start)

	maxlen = int(input("Enter maximum length:"))
	TextSliced = Text[start:end]
	palindromes = []
	starttime = time.time()
	for k in TextSliced.split():
		Index = []
		titles = []
		if(len(k)>maxlen):
			#print("t",k)
			palin = palindrome(k,maxlen)
			if(len(palin)>0): 
				#print(palin)
				#print("Story:",title,end=" ")
				for i in palin:
					palindromes.append(i)
	endtime = time.time()
	setP = set(palindromes)
	palindromes = list(setP)	
	#print((palindromes))
	for i in palindromes:
		titles = []
		print("Palindrome ->",i,"  Found at",kmpMatcher(Text,i,0,len(Text)-1))
		for key in dic.keys():
			if(i in dic[key]):
				titles.append(key)
		print("Found in",titles)
		print("---------------------------------------------------------")	
		#Index.append(Text.index(i))
		#print(Index)
	return endtime - starttime

def Build_index(Text,algo):
	#print(t)
	#Text = text.split()
	#Text = " ".join(Text)
	Text = Text.replace('"','')
	Text = Text.replace("'",'')
	Text = Text.replace(',','')
	Text = Text.replace('.','')
	Text = Text.replace('!','')
	Text = Text.replace('(','')
	Text = Text.replace(')','')
	Text = Text.replace('-','')
	Text = Text.replace('&','')
	Text = Text.replace(';','')

	text = Text
	Text = Text.split()
	Text = set(Text)
	Text = list(Text)
	Text = sorted(Text, key=str.lower)
	Numof = 0
	NumofInStory = 0
	#print(Text)
	if(algo == 1):
		for i in Text:
			print(i," -> ",end=" ")	
			#print(kmpMatcher(t,i,0,len(t)-1))
			Total = len(Rabin_Karp_Matcher(text,"\n"+i+" ",0,len(text)-1,26,97)) + len(Rabin_Karp_Matcher(text," "+i+" ",0,len(text)-1,26,97)) +len(Rabin_Karp_Matcher(text," "+i+"\n",0,len(text)-1,26,97))
			print("Number of occurences:",Total)
			print("found in:")
			for key in dic.keys():
				if(("\n"+i+" " in dic[key]) or (" "+i+" " in dic[key]) or (" "+i+"\n" in dic[key]) ):
					titles.append(key)
					NumofInStory = len(Rabin_Karp_Matcher(dic[key],"\n"+i+" ",0,len(dic[key])-1,26,97)) + len(Rabin_Karp_Matcher(dic[key]," "+i+" ",0,len(dic[key])-1,26,97)) + len(Rabin_Karp_Matcher(dic[key]," "+i+"\n",0,len(dic[key])-1,26,97))
					print(key,"--> No of occurences in story =",NumofInStory)	
			print("-------------------------------------------------------------------")
		
	if(algo == 2):
		for i in Text:
			print(i," -> ",end=" ")	
			#print(kmpMatcher(t,i,0,len(t)-1))
			Total = len(kmpMatcher(text," "+i+" ",0,len(text)-1)) + len(kmpMatcher(text,"\n"+i+" ",0,len(text)-1)) + len(kmpMatcher(text," "+i+"\n",0,len(text)-1))
			print("Number of occurences:",Total)
			print("found in:")
			for key in dic.keys():
				if(i in dic[key]):
					titles.append(key)
					NumofInStory = len(kmpMatcher(dic[key]," "+i+" ",0,len(dic[key])-1)) + len(kmpMatcher(dic[key],"\n"+i+" ",0,len(dic[key])-1)) + len(kmpMatcher(dic[key]," "+i+"\n",0,len(dic[key])-1))
					print(key,"--> No of occurences in story =",NumofInStory)	
			print("-------------------------------------------------------------------")
		
	if(algo == 3):
		for i in Text:
			print(i," -> ",end=" ")	
			#print(kmpMatcher(t,i,0,len(t)-1))
			Total = suffix_pattern(" "+i+" ",text)+ suffix_pattern("\n"+i+" ",text) + suffix_pattern(" "+i+"\n",text)
			print("Number of occurences:",Total)
			print("found in:")
			for key in dic.keys():
				if(("\n"+i+" " in dic[key]) or (" "+i+" " in dic[key]) or (" "+i+"\n" in dic[key]) ):
					titles.append(key)
					NumofInStory = suffix_pattern(" "+i+" ",dic[key])+ suffix_pattern("\n"+i+" ",dic[key]) + suffix_pattern(" "+i+"\n",dic[key])
					print(key,"--> No of occurences in story =",NumofInStory)	
			print("-------------------------------------------------------------------")

cont = "y"		
while(cont == "y"):
		
	choice = int(input(("1 -> FSA REG EX\n2 -> Pattern Searchong\n3 -> Buildindex\n4 -> Palindrome\n")))
	#print(Text.
	pr = 0
	if(choice == 1):
		inp = open("TALES.txt").read()
		li=inp.split()
		'''print(li)
		for u in li:
			print(u)'''
		tr=[]
		for t in li:
			for r in t:
				tr.append(ord(r))
		gf=set(tr)
		#print(gf)
		no=len(gf)	



		#txt = "efjsfndmnfdnhttp://dbdbn"
		pat = ["http://","ftp://","https://"]

		TF= [ [ 0 for j in range(257) ] for i in range(len(pat)+2) ]

		def getNextState(pat, M,state, x):
			#print("hi1")
			if state < M and x == ord(pat[state]):
				return state+1
			i=0
			for ns in range(state,0,-1):
		
				if ord(pat[ns-1]) == x:
					#print("hi1")
					for i in range(0,ns-1):
			
						if pat[i] != pat[state-ns+1+i]:
							break
			
			
					if i == ns-1:
						return ns
	
			return 0



		def computeTF(pat,M,TF):

			#print("hi")
			for state in range(0,M+1):
				for x in gf:	
					TF[state][x] = getNextState(pat, M, state, x)


		def search(pat1,li):
			empty=[]
	
			for gh in range(3):
				pat=pat1[gh]
				#lo=li
				M=len(pat)
		
				lo=[]
				#count=0
				for p in li:
			
					#print(p)
		
					N = len(p)
					TF= [ [ 0 for j in range(257) ] for g in range(M+2) ]


					computeTF(pat, M, TF)

					state=0
					for h in range(0,N):
						state = TF[state][ord(p[h])]
						#print(state)
						if state == M:
							#print("Pattern found at index", h-M+1)
							#print(p)
							#print(li[count])
							empty.append(p)
					
					#print(li)	
						#print(li)
					#count=count+1		
			return li,empty





		starttime = time.time()
		pq,pr=search(pat, li);
		endtime = time.time()
		Time = endtime - starttime
		timeFsa = Time
		print("infection list of url\n",len(pr))
		jam= [item for item in pq if item not in pr]
		correct=' '.join(jam)
		#print("\n",correct)
		print(pr)
		print("Time Taken =",timeFsa);

	elif(choice ==2):
		Time,algo,pattern = find_pattern();
	elif(choice == 3):
		#print(dic['FOOTNOTES'])
		#print('FOOTNOTES' in dic.keys())
		choice1 = int(input(("1 -> Rabin karp \n2 -> KMP algorithm\n3 -> Suffix Array\n")))
		start = time.time()	
		Build_index(Text,choice1)			
		end = time.time()
		timeBuild = end - start
	elif(choice == 4):
		timePalin = findPalindrome()
	print("PRINT STATS")
	#print("Infection List",pr);
	if(choice == 2):
		print("pattern ->",pattern," Algorithm Used for pattern searching pattern->",algo,"Preprocessing time->",round(pT,6),"s   ","Search Time taken->",round(Time,6),"s")
	if(choice == 3):
		print(choice1)
		if(choice1 == 1):
			print("hello")
			algo = "Rabin karp"
		elif(choice1 == 2):
			algo = "KMP algorithm"
		else:
			algo = "Suffix Array"
	
		print("Algorithm Used for Buildindex ->",algo," Time taken->",round(timeBuild,6),"s")
	if(choice == 4):
		print("time taken to find palindromes->",round(timePalin,6),"s")
	#cont = "n"
	cont = input("To continue press 'y' and to exit press n\n")
	

	
	
		

