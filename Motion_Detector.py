#Importing Libraries for motion detection
import cv2 # used for Computer Vision
import pandas as pd #Used for storing values of date and time to csv file
from datetime import datetime #Used for getting the current time
#Detection of motion
f_f = None 
s_l = [None, None]
times = []
df = pd.DataFrame(columns = ['Start', 'End']) #Creating a data structure to manupulate data
video = cv2.VideoCapture(0) #Creating a video capture object

while True:
    c, f = video.read() #Reading each frame 
    s = 0
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) #Converting each frame into grayscale
    gray = cv2.GaussianBlur(gray, (21, 21), 0) #Converting each gray frame to outlined image(Gaussian Image)
    if f_f is None:
        f_f = gray #Checking if frame is changed or not and if it's not changed then update the frame and continue with other frame 
        continue
    d_f = cv2.absdiff(f_f, gray) #If frame gets changed calculate the absolute difference between the first frame and current frame
    t_d = cv2.threshold(d_f, 30, 255, cv2.THRESH_BINARY)[1] #Creating a threshold value to compare it with current frame
    t_d = cv2.dilate(t_d, None, iterations = 0)
    (_,cnts,_) = cv2.findContours(t_d.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Making contours of the object by which the frame gets changed
    for contour in cnts:
        if cv2.contourArea(contour) < 10000: #Checking for contour area by an arbitary constant you can change it to inc./dec. sensitivity of this detector
            continue
        s = 1
        (x, y, w, h) = cv2.boundingRect(contour) #Getting the coordinates of that contour
        cv2.rectangle(f, (x, y), (x+w, y+h), (0, 255, 255), 3) #Creating rectangle box arround it(You can change the color and thickness of the box)
    s_l.append(s)
    s_l = s_l[-2:]
    if s_l[-1] == 1 and s_l[-2] == 0:
        times.append(datetime.now()) #Appending the current date and time if object leaves the frame
    if s_l[-1] == 0 and s_l[-2] == 1:
        times.append(datetime.now()) #Appending the current date and time if object enters the frame
    cv2.imshow("frame1", f) #Displaying video with current frame
    cv2.imshow("frame3", d_f) #Displayin video with outlined image of first frame
    key = cv2.waitKey(1)
    if key == ord('q'): #If you want to quit the frame then press q
        break

#Printing values of times 
print(s_l)
print(times)
a = len(times)
video.release() 
cv2.destroyAllWindows() #Releasing video capture object 

#Converting data into a csv file
for j in range(0, a, 2):
    df = df.append({'Start':times[j], 'End':times[j+1]}, ignore_index = True) #Appending data in dataframe
df.to_csv("Times.csv") #Creating a csv file

#Plotting data visualisation 
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
df["Start_string"] = df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["End_string"] = df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")
cds = ColumnDataSource(df)
p = figure(x_axis_type = 'datetime', height = 200, width = 1000, title = "Motion Graph")
p.yaxis.minor_tick_line_color = None
p.ygrid[0].ticker.desired_num_ticks = 1
hover = HoverTool(tooltips = [("Start", "@Start_string"), ("End", "@End_string")])
p.add_tools(hover)
q = p.quad(left = "Start", right = "End", bottom = 0, top = 1, color = "red", source = cds)
output_file("Graph1.html")
show(p)