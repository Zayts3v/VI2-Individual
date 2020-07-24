#include "optixParams.h" // our launch params

extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}

// ray types
enum { RAIDANCE=0, SHADOW, RAY_TYPE_COUNT };

struct RadiancePRD {
    float3   emitted;
    float3   radiance;
    float3   attenuation;
    float3   origin;
    float3   direction;
    bool done;
    uint32_t seed;
    int32_t  countEmitted;
} ;

struct shadowPRD {
    float shadowAtt;
    uint32_t seed;
} ;


extern "C" __global__ void __closesthit__radiance() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    RadiancePRD &prd = *(RadiancePRD *)getPRD<RadiancePRD>();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    const float3 nn = normalize(make_float3(n));
    // intersection position
    const float3 &rayDir =  optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir ;


    if (prd.countEmitted && length(sbtData.emission) != 0) {
        prd.emitted = sbtData.emission ;
        return;
    }
    else
        prd.emitted = make_float3(0.0f);

    uint32_t seed = prd.seed;

    float3 color;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        color = make_float3(fromTexture);
    } else {
        color = sbtData.diffuse;
    }

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere( z1, z2, w_in );
        Onb onb( nn );
        onb.inverse_transform( w_in );
        prd.direction = w_in;
        prd.origin    = pos;

        prd.attenuation *= color;
        prd.countEmitted = false;
    }
    
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;

    const float3 lightV1 = make_float3(0.47f, 0.0, 0.0f);
    const float3 lightV2 = make_float3(0.0f, 0.0, 0.38f);
    const float3 light_pos = make_float3(optixLaunchParams.global->lightPos) + lightV1 * z1 + lightV2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - pos );
    const float3 L     = normalize(light_pos - pos );
    const float  nDl   = dot( nn, L );
    const float3 Ln    = normalize(cross(lightV1, lightV2));
    const float  LnDl  = -dot( Ln, L );

    float weight = 0.0f;
    if (nDl > 0.0f && LnDl > 0.0f) {
        uint32_t occluded = 0u;
        optixTrace(optixLaunchParams.traversable,
            pos,
            L,
            0.001f,                                // tmin
            Ldist - 0.01f,                         // tmax
            0.0f,                                  // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            SHADOW,                                // SBT offset
            RAY_TYPE_COUNT,                        // SBT stride
            SHADOW,                                // missSBTIndex
            occluded);

        if (!occluded) {
            const float att = Ldist * Ldist;
            const float A = length(cross(lightV1, lightV2));
            weight = nDl * LnDl * A  / att;
        }
    }

    float probability = (sbtData.diffuse.x + sbtData.diffuse.y + sbtData.diffuse.z) / 3;
    float random = rnd(seed);

    if (random < probability) {
        prd.done = false;
    } else {
        prd.done = true;
    }

    prd.radiance += (make_float3(5.0f, 5.0f, 5.0f) * weight * optixLaunchParams.global->lightScale) / probability;
}


extern "C" __global__ void __anyhit__radiance() {

}


// miss sets the background color
extern "C" __global__ void __miss__radiance() {

    RadiancePRD &prd = *(RadiancePRD*)getPRD<RadiancePRD>();
    // set black as background color
    prd.radiance = make_float3(0.0f, 0.0f, 0.0f);
    prd.done = true;
}


// -----------------------------------------------
// Shadow rays

extern "C" __global__ void __closesthit__shadow() {

    optixSetPayload_0( static_cast<uint32_t>(true));
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow() {

    optixSetPayload_0( static_cast<uint32_t>(false));
}


// -----------------------------------------------
// Primary Rays
extern "C" __global__ void __raygen__renderFrame() {

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  

    const int &maxDepth = optixLaunchParams.frame.maxDepth;
 
    float squaredRaysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    float2 delta = make_float2(1.0f/squaredRaysPerPixel, 1.0f/squaredRaysPerPixel);

    float3 result = make_float3(0.0f);

    uint32_t seed = tea<4>( ix * optixGetLaunchDimensions().x + iy, optixLaunchParams.frame.frame );

    for (int i = 0; i < squaredRaysPerPixel; ++i) {
        for (int j = 0; j < squaredRaysPerPixel; ++j) {

            const float2 subpixel_jitter = make_float2( delta.x * (i + rnd(seed)), delta.y * (j + rnd( seed )));
            const float2 screen(make_float2(ix + subpixel_jitter.x, iy + subpixel_jitter.y)
                            / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
        
            // note: nau already takes into account the field of view and ratio when computing 
            // camera horizontal and vertical
            float3 origin = camera.position;
            float3 rayDir = normalize(camera.direction
                                + (screen.x ) * camera.horizontal
                                + (screen.y ) * camera.vertical);

            RadiancePRD prd;
            prd.emitted      = make_float3(0.f);
            prd.radiance     = make_float3(0.f);
            prd.attenuation  = make_float3(1.f);
            prd.countEmitted = true;
            prd.done         = false;
            prd.seed         = seed;

            uint32_t u0, u1;
            packPointer( &prd, u0, u1 );             
            
            for (int k = 0; k < maxDepth && !prd.done; ++k) {

                optixTrace(optixLaunchParams.traversable,
                        origin,
                        rayDir,
                        0.001f,    // tmin
                        1e20f,  // tmax
                        0.0f, OptixVisibilityMask( 1 ),
                        OPTIX_RAY_FLAG_NONE, RAIDANCE, RAY_TYPE_COUNT, RAIDANCE, u0, u1 );

                result += prd.emitted;
                result += prd.radiance * prd.attenuation;

                origin = prd.origin;
                rayDir = prd.direction;
            }
        }
    }

    result = result / (squaredRaysPerPixel*squaredRaysPerPixel);
    float gamma = optixLaunchParams.global->gamma;
    // compute index
    const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;

    optixLaunchParams.global->accumBuffer[fbIndex] = 
        (optixLaunchParams.global->accumBuffer[fbIndex] * optixLaunchParams.frame.subFrame +
        make_float4(result.x, result.y, result.z, 1)) /(optixLaunchParams.frame.subFrame+1);

    
    float4 rgbaf = optixLaunchParams.global->accumBuffer[fbIndex];
    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*min(1.0f, pow(rgbaf.x, 1/gamma)));
    const int g = int(255.0f*min(1.0f, pow(rgbaf.y, 1/gamma)));
    const int b = int(255.0f*min(1.0f, pow(rgbaf.z, 1/gamma))) ;

    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000 | (r<<0) | (g<<8) | (b<<16);
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  

extern "C" __global__ void __closesthit__phong_metal() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];
    // ray payload

    float3 normal = normalize(make_float3(n));

    // entering glass
    //if (dot(optixGetWorldRayDirection(), normal) < 0)

    RadiancePRD &prd = *(RadiancePRD*)getPRD<RadiancePRD>();

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    //(1.f-u-v) * A + u * B + v * C;
    
    const float glossiness = 20000.0f;

    float3 rayDir;
    float3 reflectDir = reflect(optixGetWorldRayDirection(), normal);
    unsigned int seed = prd.seed;

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    cosine_power_sample_hemisphere( z1, z2, rayDir, glossiness );
    Onb onb( reflectDir );
    onb.inverse_transform( rayDir );

    prd.origin = pos;
    prd.direction = rayDir;

    prd.seed = seed;
}

extern "C" __global__ void __closesthit__phong_glass() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    float3 normal = normalize(make_float3(n));
    const float3 normRayDir = optixGetWorldRayDirection();
    RadiancePRD &prd = *(RadiancePRD*)getPRD<RadiancePRD>();

    // new ray direction
    float3 refractDir;
    float3 reflectDir;

    // entering glass
    float cosTeta1;
    float teta1;
    float cosTeta2;
    float teta2;
    float n1;
    float n2;

    if (dot(normRayDir, normal) < 0) {
        n1 = 1.0;
        n2 = 1.5;
        cosTeta1 = dot(normRayDir, -normal);
        teta1 = acosf(cosTeta1);
        teta2 = asinf(((sin(teta1) * n1) / n2));
        cosTeta2 = cos(teta2);
        refractDir = refract(normRayDir, normal, 0.66);
        reflectDir = reflect(normRayDir, normal);
    }
    // exiting glass
    else {
        n1 = 1.5;
        n2 = 1.0;
        cosTeta1 = dot(normRayDir, normal);
        teta1 = acosf(cosTeta1);
        teta2 = asinf(((sin(teta1) * n1) / n2));
        cosTeta2 = cos(teta2);
        refractDir = refract(normRayDir, -normal, 1.5);
        reflectDir = reflect(normRayDir, normal);
    }

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    
    uint32_t seed = prd.seed;

    const float rand = rnd(seed);
    prd.seed = seed;

    float fr1 = pow((((n2*cosTeta1) - (n1*cosTeta2)) / ((n2*cosTeta1) + (n1*cosTeta2))), 2);
    float fr2 = pow((((n1*cosTeta2) - (n2*cosTeta1)) / ((n1*cosTeta2) + (n2*cosTeta1))), 2);

    float fr = ((fr1 + fr2) / 2.0);
    float3 outDir;

    if (fr > rand) {
        outDir = reflectDir;
    } else {
        outDir = refractDir;
    }

    prd.origin = pos;
    prd.direction = outDir;
}


extern "C" __global__ void __closesthit__shadow_glass() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    float3 normal = normalize(make_float3(n));
    const float3 normRayDir = optixGetWorldRayDirection();
    shadowPRD &prd = *(shadowPRD*)getPRD<shadowPRD>();

    // new ray direction
    float3 refractDir;
    float3 reflectDir;

    // entering glass
    float cosTeta1;
    float teta1;
    float cosTeta2;
    float teta2;
    float n1;
    float n2;

    if (dot(normRayDir, normal) < 0) {
        n1 = 1.0;
        n2 = 1.5;
        cosTeta1 = dot(normRayDir, -normal);
        teta1 = acosf(cosTeta1);
        teta2 = asinf(((sin(teta1) * n1) / n2));
        cosTeta2 = cos(teta2);
        refractDir = refract(normRayDir, normal, 0.66);
        reflectDir = reflect(normRayDir, normal);
    }
    // exiting glass
    else {
        n1 = 1.5;
        n2 = 1.0;
        cosTeta1 = dot(normRayDir, normal);
        teta1 = acosf(cosTeta1);
        teta2 = asinf(((sin(teta1) * n1) / n2));
        cosTeta2 = cos(teta2);
        refractDir = refract(normRayDir, -normal, 1.5);
        reflectDir = reflect(normRayDir, normal);
    }

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    
    uint32_t seed = prd.seed;

    const float rand = rnd(seed);
    prd.seed = seed;

    float fr1 = pow((((n2*cosTeta1) - (n1*cosTeta2)) / ((n2*cosTeta1) + (n1*cosTeta2))), 2);
    float fr2 = pow((((n1*cosTeta2) - (n2*cosTeta1)) / ((n1*cosTeta2) + (n2*cosTeta1))), 2);

    float fr = ((fr1 + fr2) / 2.0);
    float3 outDir;

    if (fr > rand) {
        outDir = reflectDir;
    } else {
        outDir = refractDir;
    }

    // ray payload
    shadowPRD afterPRD;
    afterPRD.shadowAtt = 1.0f;
    afterPRD.seed = prd.seed;
    uint32_t u0, u1;
    packPointer( &afterPRD, u0, u1 );  
    
    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        outDir,
        0.01f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1 );

    prd.shadowAtt = 0.95f * afterPRD.shadowAtt;
}