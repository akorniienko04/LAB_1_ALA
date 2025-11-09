
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import kagglehub


#PART 1 TASK 2
figure = np.array([
    [4, 4], [4, -4], [-4, -4], [-1, 0], [-4, 4], [4, 4]
])



# plt.axhline(0, color = 'red', linewidth = 1)
# plt.axvline(0, color = 'red', linewidth = 1)
# plt.plot(figure[:,0], figure[:,1])
# plt.grid(True)



#сюди тільки а б будь які додатні
def stretch(figure,a,b):
    matrix_peretvor = np.array([
        [a, 0],
        [0, b]
    ])

    new_fig = figure.copy() @ matrix_peretvor

    print(f"Matrix for stretching:\n{matrix_peretvor}")

    plt.plot(new_fig[:, 0], new_fig[:, 1])
    plt.grid(True)
    plt.show()

    return new_fig

# stretch(figure, 2, 2)

def shear(figure,a,b):
    matrix_peretvor = np.array([
        [1, b],
        [a, 1]
    ])

    print(f"Matrix for shear:\n{matrix_peretvor}")

    new_fig = figure.copy() @ matrix_peretvor

    plt.plot(new_fig[:, 0], new_fig[:, 1])
    plt.grid(True)
    plt.show()

    return new_fig

# shear(figure, 2, 2)



#сюди a б {1, -1}
def reflection(figure,a,b):
    matrix_peretvor = np.array([
        [a, 0],
        [0, b]
    ])

    new_fig = figure.copy() @ matrix_peretvor

    print(f"Matrix for reflection:\n{matrix_peretvor}")

    plt.plot(new_fig[:, 0], new_fig[:, 1])
    plt.grid(True)
    plt.show()


    return new_fig

# reflection(figure,-1,-1)


def rotation(figure, kyt):
    matrix_peretvor = np.array([
        [np.cos(kyt), -np.sin(kyt)],
        [np.sin(kyt), np.cos(kyt)]
    ])

    new_fig = figure.copy() @ matrix_peretvor

    print(f"Matrix for rotation:\n{matrix_peretvor}")

    plt.plot(new_fig[:, 0], new_fig[:, 1])
    plt.grid(True)
    plt.show()


    return new_fig


# rotation(figure, np.pi / 3)






#PART 1 TASK 2

# stretch(figure, 1.5, 0.5)
# shear(figure, 0.5, 0.2)
# rotation(figure, np.pi / 4)

# Comb 1
# step1 = stretch(figure, 1.5, 0.5)
# step2 = shear(step1, 0.5, 0.2)
# fig1 = rotation(step2, np.pi / 4)

# Comb 2
# step1 = rotation(figure, np.pi / 4)
# step2 = shear(step1, 0.5, 0.2)
# fig2 = stretch(step2, 1.5, 0.5)
#
# # Comb 3
# step1 = shear(figure, 0.5, 0.2)
# step2 = stretch(step1, 1.5, 0.5)
# fig3 = rotation(step2, np.pi / 4)



#PART 2

def read_off(filename: str):
    with open(filename, 'r') as f:
        # Перевіряємо, чи перший рядок починається з OFF
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')

        # Зчитуємо кількість вершин, граней та ребер (третє значення часто ігнорується)
        n_verts, n_faces, _ = map(int, f.readline().strip().split())

        # Зчитуємо координати всіх вершин (x, y, z)
        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]

        # Зчитуємо грані: перше число у рядку - кількість вершин грані (ігноруємо), далі індекси
        faces = [list(map(int, f.readline().strip().split()[1:])) for _ in range(n_faces)]

        # Повертаємо вершини у вигляді масиву NumPy та список граней
        return np.array(verts), faces


vertices, faces = read_off("/Users/artemkorniienko/R_for_DATA/archive/ModelNet40/laptop/test/laptop_0150.off")

# print(vertices.shape)
# print(faces [: 3])
# print(vertices)







# Функція для візуалізації OFF-моделі у вигляді сітки та точок
# (зручно бачити, як змінюється модель після трансформацій)
def plot_off(vertices, faces):
    fig = plt.figure(figsize=(8, 8))                 # створюємо вікно
    ax = fig.add_subplot(111, projection='3d')       # додаємо 3D координатну систему

    # Створюємо полігональну сітку з граней (faces) та додаємо її на графік
    mesh = Poly3DCollection([vertices[face] for face in faces],
                            alpha=0.3, edgecolor='k')   # прозорість 0.3, чорні ребра
    ax.add_collection3d(mesh)

    # Додаємо вершини як червоні точки
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2, c='r')

    # Підписуємо осі
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Автоматично масштабуємо сцену під модель
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    plt.show()  # показуємо результат


# Виклик функції для побудови OFF-моделі
# plot_off(vertices, faces)




#PART 2 TASK 3


def rotate_xy(fig, kyt):

    vertices_copy = fig.copy()

    matrix_peretvoren = np.array([
        [np.cos(kyt), -np.sin(kyt), 0],
        [np.sin(kyt),  np.cos(kyt), 0],
        [0,            0,           1]
    ])

    vertices_copy = vertices_copy @ matrix_peretvoren

    return vertices_copy


# rotated_xy = rotate_xy(vertices, np.pi / 4)
#
# plot_off(rotated_xy, faces)


def rotate_yz(fig, kyt):
    vertices_copy = fig.copy()

    matrix_peretvoren = np.array([
        [1, 0,                      0],
        [0, np.cos(kyt), -np.sin(kyt)],
        [0, np.sin(kyt),  np.cos(kyt)]
    ])

    vertices_copy = vertices_copy @ matrix_peretvoren

    return vertices_copy

# rotated_yz = rotate_yz(vertices, np.pi / 4)
#
# plot_off(rotated_yz, faces)




def rotate_xz(fig, kyt):
    vertices_copy = fig.copy()

    matrix_peretvoren = np.array([
        [np.cos(kyt), 0, np.sin(kyt)],
        [0,           1,           0],
        [-np.sin(kyt),0, np.cos(kyt)]
    ])

    vertices_copy = vertices_copy @ matrix_peretvoren

    return vertices_copy

# rotated_xz = rotate_xz(vertices, np.pi / 4)
#
# plot_off(rotated_xz, faces)



#PART 2 TASK 4

r_xy = np.array([
        [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
        [np.sin(np.pi / 6),  np.cos(np.pi / 6), 0],
        [0,            0,           1]
    ])
r_yz = np.array([
        [1, 0,                      0],
        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [0, np.sin(np.pi / 4),  np.cos(np.pi / 4)]
    ])

r_xz = np.array([
        [np.cos(np.pi / 3), 0, np.sin(np.pi / 3)],
        [0,           1,           0],
        [-np.sin(np.pi / 3),0, np.cos(np.pi / 3)]
    ])

final_matrix = r_xy @ r_yz @ r_xz

print(f"Matrixa peretvorenna:\n{final_matrix}")




rotated_xy = rotate_xy(vertices, np.pi / 6)
plot_off(rotated_xy, faces)

rotated_yz = rotate_yz(rotated_xy, np.pi / 4)
plot_off(rotated_yz, faces)

rotated_xz = rotate_xz(rotated_yz, np.pi / 3)
plot_off(rotated_xz, faces)






