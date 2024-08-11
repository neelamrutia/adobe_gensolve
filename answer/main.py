import numpy as np
import csv
import svgwrite
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull

class ShapeDetector:
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def is_circular(self, points):
        center = np.mean(points, axis=0)
        radii = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
        return np.std(radii) / np.mean(radii) < self.tolerance

    def is_linear(self, points, linear_tolerance=1e-5):
        if len(points) < 2:
            return False
        vector = points[-1] - points[0]
        cross_products = np.cross(points[1:] - points[0], vector)
        return np.all(np.abs(cross_products) < linear_tolerance)

    def is_pentagram(self, points):
        if len(points) != 5:
            return None
        
        def calculate_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        centroid = np.median(points, axis=0)
        angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        distances = []
        for i in range(5):
            distances.append(calculate_distance(sorted_points[i], sorted_points[(i + 2) % 5]))

        side = distances[0]
        return all(abs(side - dist) < self.tolerance for dist in distances)

    def is_quadrilateral(self, points):
        if len(points) != 4:
            return False

        def calculate_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        distances = []
        for i in range(4):
            for j in range(i + 1, 4):
                distances.append(calculate_distance(points[i], points[j]))

        distances.sort()

        side = distances[0]
        if not all(abs(side - distances[i]) < self.tolerance for i in range(4)):
            return None

        diagonal = distances[4]
        if not abs(diagonal - distances[5]) < self.tolerance:
            return None
        
        if not abs(diagonal - np.sqrt(2) * side) < self.tolerance:
            return None

        return True

    def is_oval(self, points):
        center = np.mean(points, axis=0)
        covariance = np.cov(points.T)
        eigenvalues, _ = np.linalg.eigh(covariance)
        aspect_ratio = np.sqrt(max(eigenvalues) / min(eigenvalues))
        return aspect_ratio > 1 + self.tolerance

    def is_curved_quadrilateral(self, points):
        if len(points) < 4:
            return False
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return not self.is_rectangular(hull_points)

    def is_rectangular(self, points):
        if len(points) != 4:
            return False

        def calculate_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        def calculate_angle(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)

        side_lengths = [calculate_distance(points[i], points[(i + 1) % 4]) for i in range(4)]
        diagonal_lengths = [calculate_distance(points[i], points[(i + 2) % 4]) for i in range(2)]

        if not (np.abs(side_lengths[0] - side_lengths[2]) < self.tolerance and 
                np.abs(side_lengths[1] - side_lengths[3]) < self.tolerance):
            return False

        if not np.abs(diagonal_lengths[0] - diagonal_lengths[1]) < self.tolerance:
            return False

        angles = [] 
        for i in range(4):
            v1 = points[(i + 1) % 4] - points[i]
            v2 = points[(i + 2) % 4] - points[(i + 1) % 4]
            angles.append(calculate_angle(v1, v2))

        return all(80 < angle < 100 for angle in angles)

    def is_symmetrical(self, points):
        if len(points) < 3:
            return False
        
        center = np.mean(points, axis=0)
        reflected_points = 2 * center - points
        
        for point in points:
            distances = np.linalg.norm(reflected_points - point, axis=1)
            min_distance = np.min(distances)
            
            if min_distance > self.tolerance:
                return False
        
        return True

    def identify_shape(self, points):
        if self.is_linear(points):
            return "line", points
        if self.is_circular(points):
            return "circle", points
        if self.is_oval(points):
            return "ellipse", points
        if len(points) == 4 and self.is_quadrilateral(points):
            return "square", points
        if len(points) == 4 and self.is_rectangular(points):
            return "rectangle", points
        if self.is_curved_quadrilateral(points):
            return "curved_rectangle", points
        if len(points) == 5 and self.is_pentagram(points):
            return "star", points
        if self.is_symmetrical(points):
            return "symmetric_shape", points
        
        return "polyline", points

class SVGCreator:
    def __init__(self, shape_detector):
        self.shape_detector = shape_detector

    def points_to_bezier(self, points, smoothness=1e-5):
        tck, u = splprep([points[:, 0], points[:, 1]], s=smoothness)
        u_new = np.linspace(0, 1, 1000)
        x_new, y_new = splev(u_new, tck)
        return np.array([x_new, y_new]).T

    def create_svg(self, all_paths, output_path):
        drawing = svgwrite.Drawing(output_path, profile='tiny')
        for path in all_paths:
            for points in path:
                shape, coords = self.shape_detector.identify_shape(points)
                if shape == "line":
                    path_data = f"M{coords[0, 0]},{coords[0, 1]} L{coords[-1, 0]},{coords[-1, 1]}"
                    drawing.add(drawing.path(d=path_data, fill='none', stroke='red', stroke_width=2))
                elif shape == "circle":
                    center = np.mean(coords, axis=0)
                    radius = np.mean(np.sqrt((coords[:, 0] - center[0])**2 + (coords[:, 1] - center[1])**2))
                    drawing.add(drawing.circle(center=(center[0], center[1]), r=radius, stroke='red', fill='none', stroke_width=2))
                elif shape == "ellipse":
                    center = np.mean(coords, axis=0)
                    covariance = np.cov(coords.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    major_axis = 2 * np.sqrt(max(eigenvalues))
                    minor_axis = 2 * np.sqrt(min(eigenvalues))
                    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[1, 0]))
                    drawing.add(drawing.ellipse(center=(center[0], center[1]), r=(major_axis / 2, minor_axis / 2), 
                                                stroke='black', fill='none', stroke_width=2, 
                                                transform=f"rotate({angle} {center[0]} {center[1]})"))
                elif shape in ["square", "rectangle", "curved_rectangle", "star", "symmetric_shape"]:
                    bezier_curve = self.points_to_bezier(coords)
                    path_data = f"M{bezier_curve[0, 0]},{bezier_curve[0, 1]} "
                    for j in range(1, len(bezier_curve)):
                        path_data += f"L{bezier_curve[j, 0]},{bezier_curve[j, 1]} "
                    stroke_color = 'blue' if shape == "curved_rectangle" else 'green' if shape == "symmetric_shape" else 'red'
                    drawing.add(drawing.path(d=path_data, fill='none', stroke=stroke_color, stroke_width=2))
                else:
                    bezier_curve = self.points_to_bezier(points)
                    path_data = f"M{bezier_curve[0, 0]},{bezier_curve[0, 1]} "
                    for j in range(1, len(bezier_curve)):
                        path_data += f"L{bezier_curve[j, 0]},{bezier_curve[j, 1]} "
                    drawing.add(drawing.path(d=path_data, fill='none', stroke='red', stroke_width=2))
        drawing.save()

class CSVReader:
    @staticmethod
    def read_csv(file_path):
        np_path_points = np.genfromtxt(file_path, delimiter=',')
        all_paths = []
        for i in np.unique(np_path_points[:, 0]):
            np_points = np_path_points[np_path_points[:, 0] == i][:, 1:]
            paths = []
            for j in np.unique(np_points[:, 0]):
                points = np_points[np_points[:, 0] == j][:, 1:]
                paths.append(points)
            all_paths.append(paths)
        return all_paths

def main():
    input_file = "./frag1.csv"  
    output_file = "./frag1.svg"  
    
    csv_reader = CSVReader()
    all_paths = csv_reader.read_csv(input_file)
    
    shape_detector = ShapeDetector()
    svg_creator = SVGCreator(shape_detector)
    svg_creator.create_svg(all_paths, output_file)

if __name__ == "__main__":
    main()