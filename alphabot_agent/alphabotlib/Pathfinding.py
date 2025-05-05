import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import json
from PIL import Image, ImageDraw

class Node:
        #Each node has connected nodes (conn), meaning an adjacent node is reachable
        #The path is used when exploring the solutions
        def __init__(self, id: int, conn:list[int] = [], path:list[int] = []):
            self.id = id
            self.conn = conn
            self.path = path

        def __str__(self):
            return str(self.id)

class Pathfinding:
    # We now assume we get the maze as a 2D array with descriptions of the walls as input
    # Array should look like  [["tl", "tb", "tr",...],[...],...]
    # letters define if there are walls (t = top, b = bottom, l = left, r = right)
    def __init__(self):
        pass

    def __arr2graph(self, maze:list[list[str]]) -> None:
        row = len(maze)
        col = len(maze[1])

        nodes: list[Node] = []
        for i, l in enumerate(maze):
            for j, walls in enumerate(l):
                conn =  []
                if (i-1) * col + j >= 0 and "t" not in walls:
                    conn.append((i-1) * col + j)
                if j > 0 and "l" not in walls:
                    conn.append(i * col + j - 1)
                if j < col-1 and "r" not in walls:
                    conn.append(i * col + j + 1)
                if (i + 1) * col < row*col -1 and "b" not in walls:
                    conn.append((i + 1) * col + j)
                nodes.append(Node(i*col+j, conn))
        self.nodes = nodes


    # BFS recursive function for path finding
    def __find_path(self, stop: int, n: list[Node], step: int = 0) -> list[int]:
        for i in n:
            if i.id == stop:
                return i.path + [i.id]

        new_n: list[Node] = []
        for i in n:
            for j in i.conn:
                if i.id != j and j not in i.path:

                    new_n.append(Node(j, self.nodes[j].conn, i.path + [i.id]))
        return self.__find_path(stop, new_n, step + 1)

    def get_path_from_maze(self, maze: list[list[str]], start: int, stop: int) -> list[int]:
        self.__arr2graph(maze)
        try:
            return self.__find_path(stop, [self.nodes[start]])
        except:
            return []

    def draw_on_pic(self, path, path2, save_image=False):
        import base64
        import io

        board_size = (11,3)
        top_left = (65,45)
        bottom_right = (1680,475)

        with Image.open("maze.jpg") as im:

            draw = ImageDraw.Draw(im)

            size = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

            square_size = size[0] / board_size[0], size[1] / board_size[1]

            for i in range(len(path[:-1])):
                start = path[i]%board_size[0] * square_size[0] + top_left[0] + square_size[0]/2, int(path[i]/board_size[0]) * square_size[1] + top_left[1] + square_size[1]/2
                stop = path[i+1]%board_size[0] * square_size[0] + top_left[0] + square_size[0]/2, int(path[i+1]/board_size[0]) * square_size[1] + top_left[1] + square_size[1]/2
                draw.line(start + stop, fill=(0,0,200), width=8)

            if len(path2) != 0:
                for i in range(len(path2[:-1])):
                    start = path2[i]%board_size[0] * square_size[0] + top_left[0] + square_size[0]/2, int(path2[i]/board_size[0]) * square_size[1] + top_left[1] + square_size[1]/2
                    stop = path2[i+1]%board_size[0] * square_size[0] + top_left[0] + square_size[0]/2, int(path2[i+1]/board_size[0]) * square_size[1] + top_left[1] + square_size[1]/2
                    draw.line(start + stop, fill=(200,0,0), width=8)

            if save_image:
                im.save("out.png")

            # Convert to base64
            buffered = io.BytesIO()
            im.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str

    def draw_maze(self, maze: list[list[str]], path: list[int] = None, path2:list[int] = None):
        rows = len(maze)
        cols = len(maze[0]) if rows > 0 else 0

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # So (0,0) is at top-left
        ax.axis('off')

        # Draw each cell's walls
        for y in range(rows):
            for x in range(cols):
                cell = maze[y][x]
                cell_number = y * cols + x
                if path:
                    if cell_number in path:
                        if path[0] == cell_number:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='blue', alpha=0.3))
                        elif path[-1] == cell_number:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='green', alpha=0.3))
                        else:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='red', alpha=0.3))
                if path2:
                    if cell_number in path2:
                        if path2[0] == cell_number:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='yellow', alpha=0.3))
                        elif path2[-1] == cell_number:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='purple', alpha=0.3))
                        else:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='pink', alpha=0.3))
                if path and path2:
                    if cell_number in path and cell_number in path2:
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, color='red', alpha=1))

                # Top wall - draw gray line first, then black if wall exists
                ax.add_line(Line2D([x, x+1], [y, y], color='gray', linewidth=0.5, alpha=0.5))
                if 't' in cell:
                    ax.add_line(Line2D([x, x+1], [y, y], color='black', linewidth=2))

                # Bottom wall
                ax.add_line(Line2D([x, x+1], [y+1, y+1], color='gray', linewidth=0.5, alpha=0.5))
                if 'b' in cell:
                    ax.add_line(Line2D([x, x+1], [y+1, y+1], color='black', linewidth=2))

                # Left wall
                ax.add_line(Line2D([x, x], [y, y+1], color='gray', linewidth=0.5, alpha=0.5))
                if 'l' in cell:
                    ax.add_line(Line2D([x, x], [y, y+1], color='black', linewidth=2))

                # Right wall
                ax.add_line(Line2D([x+1, x+1], [y, y+1], color='gray', linewidth=0.5, alpha=0.5))
                if 'r' in cell:
                    ax.add_line(Line2D([x+1, x+1], [y, y+1], color='black', linewidth=2))

        plt.savefig("in.png")

    def avoid_collision(self, curr_path, other_path, margin=1):
        # We give priority to the longest path
        hasPrio = len(curr_path) > len(other_path)

        for i in range(0, len(curr_path)):
            # Bounding the index to other path length to avoid out of bound, but still check as if the robot was stopped at his target
            other_i = min(i, len(other_path) - margin - 1)

            curr_indexes = curr_path[i - margin:i + margin]
            other_indexes = other_path[other_i - margin:other_i + margin]

            for j, n in enumerate(curr_indexes):
                if n in other_indexes:
                    # We have a collision :(
                    # We need to move the end of the path of the robot who has not the priority
                    if hasPrio:
                        res_other = other_path[:i-margin+j]
                        res = curr_path
                        print("I have prio, I'm reducing the other path")
                    else:
                        res_other = other_path
                        res = curr_path[:i-margin+j]
                        print("I don't have prio, I'm reducing my path")
                    return res, res_other

        return curr_path, other_path

    def problem_detect(self, path1, path2):
        paths = (path1, path2) if len(path1) < len(path2) else (path2, path1)

        for idx, i in enumerate(paths[0][0:-1]):
            if i == paths[1][idx]:
                return idx - 1
            elif i == paths[1][idx + 1]:
                if paths[1][idx] == paths[0][idx + 1]:
                    return idx
        return False

    def get_json_from_path(self, path, towards=0):
        prev = ""
        orientation = ""
        ori_target = 0
        inst = []
        for idx, i in enumerate(path[:-1]):
            if path[idx + 1] == i + 1:
                orientation = "r"
                ori_target = 90
            elif path[idx + 1] == i - 1:
                orientation = "l"
                ori_target = 270
            elif path[idx + 1] == i - 11:
                orientation = "u"
                ori_target = 0
            elif path[idx + 1] == i + 11:
                orientation = "d"
                ori_target = 180
            else:
                print("uh oh")
            if prev == "":
                init_rota = ori_target - towards
                init_rota = init_rota - 360 if init_rota > 180 else init_rota
                init_rota = init_rota + 360 if init_rota < -180 else init_rota
                # Setting up known rotation when first launching the pathfinding
                if init_rota != 0: inst.append(f"rotate:{init_rota}")
                inst.append("forward:1")
            else:
                if prev == orientation:
                    inst.append("forward:1")
                else:
                    if prev + orientation in ["rl", "lr", "ud", "du"]:
                        inst.append("rotate:180")
                    elif prev + orientation in ["rd", "dl", "lu", "ur"]:
                        inst.append("rotate:90")
                    elif prev + orientation in ["dr", "ld", "ul", "ru"]:
                        inst.append("rotate:-90")

                    inst.append("forward:1")
            prev = orientation
        out = {"commands":[]}
        for i in inst:
            cmd = i.split(":")[0]
            arg = i.split(":")[1]
            out["commands"].append({"command":cmd, "args":[arg]})
        return out

    def get_json_from_maze(self, maze: list[list[str]], start: int, stop: int, save: bool = False, towards: int = 0):
        path = self.get_path_from_maze(maze, start, stop)
        out = self.get_json_from_path(path, towards)
        if save:
            with open("out.json", "w") as f:
                json.dump(out, f)

        return out

    def divide_path(self, path: list[int], base_maze: list[list[str]]) -> None:
        new_maze = []
        for i in path:
            if self.nodes[i] not in new_maze:
                new_maze.append(self.nodes[i])
            for j in self.nodes[i].conn:
                if self.nodes[j] not in new_maze:
                    new_maze.append(self.nodes[j])

        spot_x = min([i.id % 11 for i in new_maze])
        size_x = max([i.id % 11 for i in new_maze]) - spot_x + 1

        spot_y = min([int(i.id /11) for i in new_maze])
        size_y = max([int(i.id /11) for i in new_maze]) - spot_y + 1

        data = (spot_x, spot_y, size_x, size_y)

        maze2 = []
        for i in range(2 * size_y):
            maze2.append([])
            for j in range(2 * size_x):
                maze2[i].append("")


        for i in range(size_y):
            for j in range(size_x):
                if (i + spot_y)*11 + j + spot_x not in [q.id for q in new_maze]: # If there is not path to this square
                    maze2[i*2][j*2] += "tl"        # We lock the square
                    maze2[i*2+1][j*2] += "bl"
                    maze2[i*2][j*2+1] += "tr"
                    maze2[i*2+1][j*2+1] += "br"
                else:
                    if i  == 0:
                        maze2[i*2][j*2] += "t"
                        maze2[i*2][j*2+1] += "t"
                    elif "t" in base_maze[i + spot_y][j+spot_x] or (i + spot_y-1)*11 + j + spot_x not in [q.id for q in new_maze]:
                        maze2[i*2][j*2] += "t"
                        maze2[i*2][j*2+1] += "t"

                    if i == size_y -1:
                        maze2[i*2+1][j*2] += "b"
                        maze2[i*2+1][j*2+1] += "b"
                    elif "b" in base_maze[i + spot_y][j+spot_x] or (i + spot_y+1)*11 + j + spot_x not in [q.id for q in new_maze]:
                        maze2[i*2+1][j*2] += "b"
                        maze2[i*2+1][j*2+1] += "b"

                    if j == 0:
                        maze2[i*2][j*2] += "l"
                        maze2[i*2+1][j*2] += "l"
                    elif "l" in base_maze[i + spot_y][j+spot_x] or (i + spot_y)*11 + j + spot_x -1 not in [q.id for q in new_maze]:
                        maze2[i*2][j*2] += "l"
                        maze2[i*2+1][j*2] += "l"

                    if j == size_x -1:
                        maze2[i*2][j*2+1] += "r"
                        maze2[i*2+1][j*2+1] += "r"
                    elif "r" in base_maze[i + spot_y][j+spot_x] or (i + spot_y)*11 + j + spot_x + 1 not in [q.id for q in new_maze]:
                        maze2[i*2][j*2+1] += "r"
                        maze2[i*2+1][j*2+1] += "r"

        return maze2
