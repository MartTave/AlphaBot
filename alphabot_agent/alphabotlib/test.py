def get_corr_angle_dist(robot, next_x, next_y):
        import math
        rx = robot[2]
        ry = robot[3]

        dx = next_x-rx
        dy = next_y-ry

        try:
            angle_aligned = - math.degrees(math.atan(dx/dy))
        except:
              angle_aligned = 90 if next_x > rx else -90
        correcting_angle = angle_aligned - robot[1]

        return correcting_angle


top_left = (30, 30)
bottom_right = (250, 60)

w = (bottom_right[0] - top_left[0]) / 11
h = (bottom_right[1] - top_left[1]) / 3

curr_path = [12, 11]

next_x = (curr_path[1] % 11) * w + (w/2) + top_left[0]
next_y = int(curr_path[1] / 11) * h + (h/2) + top_left[1]

print(next_x)
print(next_y)


print(get_corr_angle_dist([0, 0, 45, 45], next_x, next_y))

