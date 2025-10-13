from world import CafeWorld

def main():
    # Create and visualize the cafe world
    cafe = CafeWorld()

    print(f"\nTotal walkable cells: {len(cafe.walkable_cells)}")
    print(f"Total sittable chairs: {len(cafe.zones['chairs'])}")
    print(f"Grid dimensions: {cafe.height} x {cafe.width}")

    cafe.print_layout()




if __name__ == "__main__":
    main()
    