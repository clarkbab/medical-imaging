import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
import SimpleITK as sitk



def register_images(fixed_image, moving_image):
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    registration_method = sitk.ImageRegistrationMethod()

    # Set the similarity metric to mean squares
    registration_method.SetMetricAsMeanSquares()

    # Set the optimizer to gradient descent
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

    # Set the interpolator to linear
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Add observer to print out updates during optimization
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: print(f"Step: {registration_method.GetOptimizerIteration()}, Metric: {registration_method.GetMetricValue()}"))

    # Execute the registration
    transform = registration_method.Execute(fixed_image, moving_image)

    # Transform the moving image
    registered_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Convert the registered image to a numpy array
    registered_array = sitk.GetArrayFromImage(registered_image)

    return registered_arra

    def plot_images(fixed_image, moving_image, transformed_image):
        # Plot the fixed image
        plt.subplot(131)
        plt.imshow(fixed_image, cmap='gray')
        plt.title('Fixed Image')
        plt.axis('off')

        # Plot the moving image
        plt.subplot(132)
        plt.imshow(moving_image, cmap='gray')
        plt.title('Moving Image')
        plt.axis('off')

        # Plot the transformed image
        plt.subplot(133)
        plt.imshow(transformed_image, cmap='gray')
        plt.title('Transformed Image')
        plt.axis('off')

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    registration_method = sitk.ImageRegistrationMethod()

    # Set the similarity metric to mean squares
    registration_method.SetMetricAsMeanSquares()

    # Set the optimizer to gradient descent
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

    # Set the interpolator to linear
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Execute the registration
    transform = registration_method.Execute(fixed_image, moving_image)

    # Transform the moving image
    registered_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Convert the registered image to a numpy array
    registered_array = sitk.GetArrayFromImage(registered_image)

    return registered_array

    import matplotlib.pyplot as plt

def plot_images(fixed_image, moving_image, transformed_image):
    # Convert SimpleITK images to numpy arrays
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(moving_image)
    transformed_array = sitk.GetArrayFromImage(transformed_image)

    # Plot the fixed image
    plt.subplot(131)
    plt.imshow(fixed_array, cmap='gray')
    plt.title('Fixed Image')
    plt.axis('off')

    # Plot the moving image
    plt.subplot(132)
    plt.imshow(moving_array, cmap='gray')
    plt.title('Moving Image')
    plt.axis('off')

    # Plot the transformed image
    plt.subplot(133)
    plt.imshow(transformed_array, cmap='gray')
    plt.title('Transformed Image')
    plt.axis('off')

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


def load_mhd_image(file_path):
    try:
        image = sitk.ReadImage(file_path)
        return image
    except RuntimeError as e:
        print(f"Error loading image: {e}")
        return None
    return image
