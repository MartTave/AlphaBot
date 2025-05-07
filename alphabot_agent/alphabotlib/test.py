def get_corr_angle_dist(robot, next_x, next_y, fact):
    import math
    rx = robot[2]
    ry = robot[3]

    dx = next_x - rx
    dy = next_y - ry

    # Standard angle: 0° is along positive X, rotate it -90° to align 0° with -Y
    angle_to_target = math.degrees(math.atan2(dy, dx)) + 90

    # correcting_angle = angle difference between where the robot is facing and where it should face
    correcting_angle = angle_to_target - robot[1]

    # Normalize again to [-180, 180]
    if correcting_angle > 180:
        correcting_angle -= 360
    elif correcting_angle < -180:
        correcting_angle += 360

    hyp = math.sqrt(dx*dx + dy*dy)
    dist = hyp / fact

    return correcting_angle, dist


top_left = (30, 30)
bottom_right = (140, 60)

w = abs(top_left[0] - bottom_right[0]) / 11
h = abs(top_left[1] - bottom_right[1]) / 3

curr_path = [12, 1]

next_x = (curr_path[1] % 11) * w + (w/2) + top_left[0]
next_y = int(curr_path[1] / 11) * h + (h/2) + top_left[1]

print(next_x)
print(next_y)

factor = w

angle = get_corr_angle_dist([0, 10, 45, 45], next_x, next_y, factor)

print(angle)