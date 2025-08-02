import Foundation
import ARKit
import CoreML

/// Simple representation of a 3D bounding box detected in the scene.
public struct BoundingBox {
    public let center: vector_float3
    public let size: vector_float3
    public let orientation: simd_quatf
}

/// Detector using DepthAnythingV2 fused with LiDAR to produce 3D bounding boxes.
///
/// This is a stub demonstrating how CoreML 4.0 could be integrated. The actual
/// DepthAnythingV2 model is not bundled in the repository.
final class BoundingBoxDetector {
    private let model: MLModel?

    init() {
        // Attempt to load the compiled DepthAnythingV2 CoreML model. The model
        // would leverage new CoreML 4.0 features such as on-device fusion and
        // flexible compute units.
        if let url = Bundle.main.url(forResource: "DepthAnythingV2", withExtension: "mlmodelc") {
            let config = MLModelConfiguration()
            // Use all available compute units (CoreML 4 feature).
            config.computeUnits = .all
            self.model = try? MLModel(contentsOf: url, configuration: config)
        } else {
            self.model = nil
        }
    }

    /// Detect bounding boxes for obstacles in the current AR frame.
    ///
    /// - Parameter frame: The ARFrame containing the latest LiDAR and image data.
    /// - Returns: An array of detected bounding boxes. Currently empty until the
    ///            model and post-processing are implemented.
    func detectBoundingBoxes(frame: ARFrame) -> [BoundingBox] {
        guard let model = model,
              let depthMap = frame.sceneDepth?.depthMap else {
            return []
        }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)

        let features: [String: MLFeatureValue] = [
            "image": MLFeatureValue(pixelBuffer: frame.capturedImage),
            "depth": MLFeatureValue(pixelBuffer: depthMap)
        ]

        guard let provider = try? MLDictionaryFeatureProvider(dictionary: features),
              let result = try? model.prediction(from: provider),
              let boxesArray = result.featureValue(for: "boxes")?.multiArrayValue else {
            return []
        }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        let depthBase = unsafeBitCast(CVPixelBufferGetBaseAddress(depthMap), to: UnsafeMutablePointer<Float32>.self)
        let depthStride = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float32>.size

        let ptr = boxesArray.dataPointer.bindMemory(to: Float32.self, capacity: boxesArray.count)
        let boxCount = boxesArray.count / 4

        let fx = frame.camera.intrinsics[0][0]
        let fy = frame.camera.intrinsics[1][1]
        let cx = frame.camera.intrinsics[2][0]
        let cy = frame.camera.intrinsics[2][1]

        var detected: [BoundingBox] = []
        for i in 0..<boxCount {
            let x = ptr[i * 4]
            let y = ptr[i * 4 + 1]
            let w = ptr[i * 4 + 2]
            let h = ptr[i * 4 + 3]

            let px = Int(x * Float(width))
            let py = Int(y * Float(height))
            guard px >= 0 && px < width && py >= 0 && py < height else { continue }

            let depth = depthBase[py * depthStride + px]
            let xn = (Float(px) - cx) / fx
            let yn = (Float(py) - cy) / fy
            let cameraCoord = SIMD3<Float>(xn * depth, yn * depth, depth)
            let world = simd_mul(frame.camera.transform, SIMD4<Float>(cameraCoord, 1.0))

            let center = vector_float3(world.x, world.y, world.z)
            let size = vector_float3(w * depth, h * depth, depth * 0.1)
            let orientation = simd_quatf(frame.camera.transform)

            detected.append(BoundingBox(center: center, size: size, orientation: orientation))
        }

        return detected
    }
}

