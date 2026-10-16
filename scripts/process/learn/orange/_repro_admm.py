import os
import time
import itk
from itk import RTK as rtk

dir_out = r"D:\Brett\code\Tomosynthesis\orange_outputs_5"

CpuImageType = itk.Image[itk.F, 3]
ImageType = itk.CudaImage[itk.F, 3]

geom_reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
geom_reader.SetFilename(os.path.join(dir_out, "geo.xml"))
geom_reader.GenerateOutputInformation()
geometry = geom_reader.GetOutputObject()
print("n angles:", len(geometry.GetGantryAngles()))

recon_size = [540, 270, 350]
recon_spacing = [1.0, 1.0, 1.0]
recon_origin = [
    -(recon_size[0] * recon_spacing[0]) / 2,
    -(recon_size[1] * recon_spacing[1]) / 2,
    -(recon_size[2] * recon_spacing[2]) / 2,
]

proj_dir = os.path.join(dir_out, "proj")
proj_files = sorted(os.path.join(proj_dir, f) for f in os.listdir(proj_dir))
print("n proj files:", len(proj_files))

proj_stack = rtk.ProjectionsReader[CpuImageType].New()
proj_stack.SetFileNames(proj_files)
proj_stack.Update()
proj_cpu = proj_stack.GetOutput()
print("proj_cpu buffered region:", proj_cpu.GetBufferedRegion())

proj_gpu = ImageType.New()
proj_gpu.SetPixelContainer(proj_cpu.GetPixelContainer())
proj_gpu.CopyInformation(proj_cpu)
proj_gpu.SetBufferedRegion(proj_cpu.GetBufferedRegion())
proj_gpu.SetRequestedRegion(proj_cpu.GetRequestedRegion())
proj_gpu.ReleaseDataFlagOff()
print("proj_gpu buffered region:", proj_gpu.GetBufferedRegion())
print("proj_gpu release data flag:", proj_gpu.GetReleaseDataFlag())

recon_source = rtk.ConstantImageSource[ImageType].New()
recon_source.SetSize(recon_size)
recon_source.SetSpacing(recon_spacing)
recon_source.SetOrigin(recon_origin)
recon_source.SetConstant(0.0)

admm = rtk.ADMMTotalVariationConeBeamReconstructionFilter[ImageType].New()
admm.SetInput(0, recon_source.GetOutput())
admm.SetInput(1, proj_gpu)
admm.SetGeometry(geometry)
admm.SetAlpha(0.03)
admm.SetBeta(10.0)
admm.SetAL_iterations(10)
admm.SetCG_iterations(5)
admm.SetForwardProjectionFilter(admm.ForwardProjectionType_FP_CUDARAYCAST)
admm.SetBackProjectionFilter(admm.BackProjectionType_BP_CUDAVOXELBASED)
admm.SetDisableDisplacedDetectorFilter(True)

start = time.perf_counter()
print("Running iterative method")
admm.Update()
print("Completed")
print(f"Elapsed: {time.perf_counter() - start:.3f} s")
