import cv2 as cv

def whitePrecentage(path):
    img = cv.imread(path)

    height, width = img.shape[:2]
    white = 0
    for y in range(height):
        for x in range(width):
            if img[y][x][0] != 0 and img[y][x][1] != 0 and img[y][x][2] != 0 :
                white += 1
    
    return((white*100)/(width*height))

def main():
    print(whitePrecentage("C:/Users/fabri/Desktop/scuola/project/fiumi_laghi/code/data/elab_jom_image3.jpg"))

if __name__ == "__main__":
    main()