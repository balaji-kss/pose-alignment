
def mapNum(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def interp_color(angle, bounds, start_color, end_color):
    interpolate = mapNum(angle, bounds[0], bounds[1], 0, 1)
    new_color = (int(start_color[0] * (1 - interpolate) + end_color[0] * interpolate), \
                 int(start_color[1] * (1 - interpolate) + end_color[1] * interpolate), \
                 int(start_color[2] * (1 - interpolate) + end_color[2] * interpolate))
    return new_color

def is_angle_in_range(range, angle):
    
    return angle >= range[0] and angle < range[1]

def get_color(joint_name, angle_bounds, angle, colors):

    for i, r in enumerate(angle_bounds[joint_name]):
        if (is_angle_in_range(r, angle)):
            color_obj = colors[i]
            if len(color_obj) == 1:
                return color_obj[0]
            else:
                start = color_obj[0]
                end = color_obj[1]
                bounds = r
                return interp_color(angle, bounds, start, end)
            
    return (0, 255, 0)