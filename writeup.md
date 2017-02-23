# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

### Reflection

### 1. My pipeline consisted of these steps:

1) Converted image to grayscale

![Grayscale Image](/gray.jpg?raw=true "Gray Scale")

2) Gaussian blurred it

![Gaussian Image](/gaussian.jpg?raw=true "Gaussian Blurred")

3) Ran Canny edge detection algorithm on it

![Edge Detection Image](/edges.jpg?raw=true "Edges After Canny Edge Detection")

4) Created a polygon region to remove areas from edge detection that are not useful for lane detection

![Masked Image](/masked.jpg?raw=true "Masked out areas of non interest")

5) Ran Hough transformation for feature extraction

![Hough on top of Edge](/edge_hough.jpg?raw=true "Hough on top of Canny")

![Hough on top of Original Image](/org_hough.jpg?raw=true "Hough on top of original image")

6) Draw lane lines for left and right lanes by averaging slope and extrapolating the line segements in each lane

![Lane Marking](/lanes.jpg?raw=true "Lane Markings")


### 2. Draw Line function

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:

#### Classification of line segments into left / right lanes
Out of the lines given by Hough transform, figure out which ones will go to the left lane and which ones will go to right lanes

```
    size_y, size_x,_ = img.shape
    center_x = size_x / 2
  
    left_lines = lines[lines[:,0,0] <= center_x]
    
    right_lines = lines[lines[:,0,2] > center_x]

```

#### Figure out an averaged lane line

In both left and right line segments, figured out an equation of the line that has the weighted average slope of all smaller line segments. The bigger a line segment, the bigger its weight in the average.

For computing how big a line is, used the line distance formula without the square root (as an optimization) since the referential weights would remain the same:

Distance formula: √((x2 - x1)<sup>2</sup> + (y2 - y1)<sup>2</sup>
Distance formula used for weighted average: ((x2 - x1)<sup>2</sup> + (y2 - y1)<sup>2</sup>

```
def avg_lane_equation(lines):
    slopes = ((lines[:,0,3] - lines[:,0,1]) / (lines[:,0,2] - lines[:,0,0]))
    distance = np.square(lines[:,0,2] - lines[:,0,0]) + np.square(lines[:,0,3] - lines[:,0,1])
    slope_avg = np.average(slopes, weights = distance)
    
    biggest_line_segment = np.argmax(distance)
    
    # C = Y - MX
    C = lines[biggest_line_segment,0,3] - slope_avg * lines[biggest_line_segment,0,2]
    return (slope_avg, C)

```

The constant **C in Y = MX + C**, is found using the biggest line segment as a reference in equation:

C = Y - MX

#### Extrapolate lane lines

Once an averaged slope and C has been computed, left and right lane lines are extrapolated using constant Y and figuring out the X's.

**Using M and C from left lane equation:**

Bottom left lane found using:

Y = Height of frame (540)
X = (Y - C) / M

Top of left lane found using:

Y = Height of frame * 0.6 (324)
X = (Y - C) / M

**Using M and C from right lane equation:**

Bottom right lane found using:

Y = Height of frame (540)
X = (Y - C) / M

Top of right lane found using:

Y = Height of frame * 0.6 (324)
X = (Y - C) / M

###2. Identify potential shortcomings with your current pipeline

A potential short comming would be curved lane lines since there might be a chance that a left lane is turning and might end up on the right side of the frame. Where as the alogrithm initially breaks left and right lane lines using the middle of the frame.

###3. Suggest possible improvements to your pipeline

Use some kind of kernel to identify which line segments from hough transformation go to left and right lanes. 