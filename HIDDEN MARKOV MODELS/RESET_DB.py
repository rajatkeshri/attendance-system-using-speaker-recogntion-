#TO RESET THE DATABASE TO 0 ATTENDANCE
import pandas as pd


def reset():

    list1=[]
    
    data = pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv") 

    names=data["NAMES"]
    names=list(names)

    usn=data["ROLL_NUMBER"]
    usn=list(usn)

    att=data["ATTENDANCE"]
    att=list(att)
    att1=list(att)
    print(att)

    rows=len(names)

    att= [0] * len(names)

    list1.append(att)
    with open('registry.txt', 'a') as f:
        for item in att1:
            print(item)
            f.write("%s" % item)
        f.write("\n")
    
    df = pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv") 
    #rg=pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/registry.csv")
    #rg=pd.DataFrame[["nishant","padma","rajat","shreekar","shruthi"]]
   

    df = df[['NAMES','ROLL_NUMBER','ATTENDANCE']]
    d=df["ATTENDANCE"]
    #rg.append(d)
    
  

    

    
    
    for i in range(0,len(names)):
        (df['ATTENDANCE'][i])=att[i]
    print("\n RESET ")
    print(df,"\n")
    df.to_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv")
    
reset()
