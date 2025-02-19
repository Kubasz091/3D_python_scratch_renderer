from objects import Objecs3D, Camera, ObjectsToRender

objects = ObjectsToRender()
cube1 = Objecs3D(0, 0, 0, "cube", (1, 1, 1))
cube2 = Objecs3D(2, 0, 0, "cube", (1, 2, 1))
cube3 = Objecs3D(0, 0, 3, "cube", (1, 1, 2))
cube4 = Objecs3D(0, 10, 0, "cube", (1, 1, 1))

objects.add(cube1)
objects.add(cube2)
objects.add(cube3)
objects.add(cube4)

print('There are ', len(objects.objects), " objects being rendered")

camera = Camera(0, -2, 0, 1.5, (2, 1.5), 2)

camera.display(objects)