class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            return
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node

    def print_list(self):
        if self.head is None:
            print("The list is empty.")
            return
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

    def delete(self, pos):
        try:
            if self.head is None:
                raise Exception("List is empty.")
            if pos <= 0:
                raise Exception("Invalid position.")
            if pos == 1:
                self.head = self.head.next
                return
            temp = self.head
            count = 1
            while temp and count < pos - 1:
                temp = temp.next
                count += 1
            if temp is None or temp.next is None:
                raise Exception("Position out of range.")
            temp.next = temp.next.next
        except Exception as e:
            print("Error:", e)

my_list = LinkedList()
my_list.add(10)
my_list.add(20)
my_list.add(30)
my_list.add(40)
my_list.add(50)
print("Original list:")
my_list.print_list()
print("Deleting node at position 3:")
my_list.delete(3)
my_list.print_list()
print("Deleting node at position 10:")
my_list.delete(10)
print("Deleting all nodes:")
for i in range(5):
    my_list.delete(1)
my_list.print_list()
my_list.delete(1)
