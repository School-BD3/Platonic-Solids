#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

class MeshBuilder {
public:
    std::vector<Vertex> vertices;

    void addTriangle(glm::vec3 v0,
                 glm::vec3 v1,
                 glm::vec3 v2)
    {
        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        glm::vec3 center = (v0 + v1 + v2) / 3.0f;

        if (glm::dot(normal, center) < 0.0f) {
            std::swap(v1, v2);
        }

        vertices.push_back({v0, glm::vec3(0.0f)});
        vertices.push_back({v1, glm::vec3(0.0f)});
        vertices.push_back({v2, glm::vec3(0.0f)});
    }
};

class Mesh {
public:
    Mesh(const std::vector<Vertex>& vertices) {
        vertexCount = static_cast<GLsizei>(vertices.size());

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glBufferData(GL_ARRAY_BUFFER,
                     vertices.size() * sizeof(Vertex),
                     vertices.data(),
                     GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              sizeof(Vertex),
                              (void*)offsetof(Vertex, position));
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                              sizeof(Vertex),
                              (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    ~Mesh() {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
    }

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Mesh(Mesh&& other) noexcept {
        vao = other.vao;
        vbo = other.vbo;
        vertexCount = other.vertexCount;

        other.vao = 0;
        other.vbo = 0;
        other.vertexCount = 0;
    }

    Mesh& operator=(Mesh&& other) noexcept {
    if (this != &other) {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);

        vao = other.vao;
        vbo = other.vbo;
        vertexCount = other.vertexCount;

        other.vao = 0;
        other.vbo = 0;
        other.vertexCount = 0;
    }
    return *this;
    }

    void draw() const {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);
    }

private:
    GLuint vao = 0;
    GLuint vbo = 0;
    GLsizei vertexCount = 0;
};

const int WIDTH = 800;
const int HEIGHT = 600;

struct ShaderUniforms {
    GLint mvp;
    GLint model;
    GLint normal;
    GLint light;
    GLint lightPos2;
    GLint lightPos3;
    GLint view;
    GLint lightSpaceMatrix;
    GLint shadowMap;
};

class ShaderManager {
public:
    static ShaderManager& getInstance() {
        static ShaderManager instance;
        return instance;
    }

    const ShaderUniforms& getUniforms(GLuint program) const {
        auto it = uniformCache.find(program);
        if (it == uniformCache.end()) {
            throw std::runtime_error("Uniforms not found for shader");
        }
        return it->second;
    }

    GLuint loadShader(const std::string& name, const char* vertexSource, const char* fragmentSource) {
        auto it = programs.find(name);
        if (it != programs.end()) {
            return it->second;
        }

        const GLuint program = createShaderProgram(vertexSource, fragmentSource);
        programs[name] = program;
        ShaderUniforms u;
        u.mvp   = glGetUniformLocation(program, "uMVP");
        u.model = glGetUniformLocation(program, "uModel");
        u.normal= glGetUniformLocation(program, "uNormalMatrix");
        u.light = glGetUniformLocation(program, "lightPos");
        u.lightPos2 = glGetUniformLocation(program, "lightPos2");
        u.lightPos3 = glGetUniformLocation(program, "lightPos3");
        u.view  = glGetUniformLocation(program, "viewPos");
        u.lightSpaceMatrix = glGetUniformLocation(program, "lightSpaceMatrix");
        u.shadowMap = glGetUniformLocation(program, "shadowMap");

        uniformCache[program] = u;

        return program;
    }

    bool hasShader(const std::string& name) const {
        return programs.find(name) != programs.end();
    }

    GLuint getShader(const std::string& name) const {
        auto it = programs.find(name);
        if (it == programs.end()) {
            throw std::runtime_error("Requested shader is not loaded: " + name);
        }
        return it->second;
    }

    void releaseAll() {
        for (const auto& entry : programs) {
            if (entry.second != 0) {
                glDeleteProgram(entry.second);
            }
        }
        programs.clear();
        uniformCache.clear();
    }

private:
    ShaderManager() = default;
    ~ShaderManager() {
        releaseAll();
    }
    ShaderManager(const ShaderManager&) = delete;
    ShaderManager& operator=(const ShaderManager&) = delete;

    GLuint compileShader(GLenum shaderType, const char* source) {
        GLuint shader = glCreateShader(shaderType);
        if (shader == 0) {
            throw std::runtime_error("Failed to create shader object");
        }

        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        int success = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
            const std::string typeName = (shaderType == GL_VERTEX_SHADER) ? "vertex" : "fragment";
            const std::string error = "Shader compile error (" + typeName + "): " + std::string(infoLog);
            glDeleteShader(shader);
            throw std::runtime_error(error);
        }

        return shader;
    }

    GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
        GLuint vertexShader = 0;
        GLuint fragmentShader = 0;
        GLuint program = 0;
        try {
            vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
            fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

            program = glCreateProgram();
            if (program == 0) {
                throw std::runtime_error("Failed to create shader program object");
            }
            glAttachShader(program, vertexShader);
            glAttachShader(program, fragmentShader);
            glLinkProgram(program);

            int success = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &success);
            if (!success) {
                char infoLog[512];
                glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
                throw std::runtime_error(std::string("Program link error: ") + infoLog);
            }

            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);
            return program;
        } catch (...) {
            if (vertexShader != 0) {
                glDeleteShader(vertexShader);
            }
            if (fragmentShader != 0) {
                glDeleteShader(fragmentShader);
            }
            if (program != 0) {
                glDeleteProgram(program);
            }
            throw;
        }
    }

    std::map<std::string, GLuint> programs;
    std::map<GLuint, ShaderUniforms> uniformCache;
};

enum class ShaderType {
    BLINN_PHONG,
    SHADOW_PASS,
    WIREFRAME,
    GRID,
    UI
};

class IShaderProvider {
public:
    virtual ~IShaderProvider() = default;
    virtual GLuint getShader(ShaderType type) = 0;
};

class ShaderFactory {
public:
    static GLuint getShader(ShaderType type) {
        ShaderManager& manager = ShaderManager::getInstance();
        const std::string name = shaderName(type);
        if (manager.hasShader(name)) {
            return manager.getShader(name);
        }
        return manager.loadShader(name, vertexSource(type), fragmentSource(type));
    }

private:
    static std::string shaderName(ShaderType type) {
        switch (type) {
            case ShaderType::BLINN_PHONG:
                return "blinn_phong";
            case ShaderType::SHADOW_PASS:
                return "shadow_pass";
            case ShaderType::WIREFRAME:
                return "wireframe";
            case ShaderType::GRID:
                return "grid";
            case ShaderType::UI:
                return "ui";
            default:
                throw std::runtime_error("Unknown shader type");
        }
    }

    static const char* vertexSource(ShaderType type) {
        switch (type) {
            case ShaderType::GRID:
    return R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 uMVP;

void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";
            case ShaderType::BLINN_PHONG:
                return R"(
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;
uniform mat4 lightSpaceMatrix;

out vec3 FragPos;
out vec3 Normal;
out vec4 FragPosLightSpace;

void main()
{
    FragPos = vec3(uModel * vec4(aPos, 1.0));
    Normal = normalize(uNormalMatrix * aNormal);

    FragPosLightSpace =
        lightSpaceMatrix * vec4(FragPos, 1.0);

    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";
            case ShaderType::SHADOW_PASS:
                return R"(
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec4 FragPosLightSpace;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D shadowMap;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords =
        fragPosLightSpace.xyz / fragPosLightSpace.w;

    projCoords =
        projCoords * 0.5 + 0.5;

    if (projCoords.z > 1.0)
        return 0.0;

    float currentDepth =
        projCoords.z;

    float bias =
        max(0.02 * (1.0 - dot(normal, lightDir)), 0.005);

    vec2 texelSize =
        1.0 / textureSize(shadowMap, 0);

    float shadow = 0.0;

    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float pcfDepth =
                texture(
                    shadowMap,
                    projCoords.xy + vec2(x, y) * texelSize
                ).r;

            shadow +=
                currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }

    shadow /= 9.0;

    return shadow;
}

void main()
{
    vec3 baseColor = vec3(0.9, 0.4, 0.2);
    vec3 ambient = 0.2 * baseColor;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);

    float diff =
        max(dot(norm, lightDir), 0.0);

    vec3 diffuse =
        diff * baseColor;

    vec3 viewDir =
        normalize(viewPos - FragPos);

    vec3 halfwayDir =
        normalize(lightDir + viewDir);

    float spec =
        pow(max(dot(norm, halfwayDir), 0.0), 32.0);

    vec3 specular =
        0.35 * spec * vec3(1.0);

    float shadow =
    ShadowCalculation(
        FragPosLightSpace,
        norm,
        lightDir
    );

    vec3 lighting =
        ambient + (1.0 - shadow) * (diffuse + specular);

    FragColor = vec4(lighting, 1.0);
}
)";
            case ShaderType::WIREFRAME:
                return R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 uMVP;
        void main()
        {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
    )";
            case ShaderType::UI:
return R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;

out vec2 TexCoord;

uniform mat4 projection;
uniform vec2 offset;
uniform vec2 size;

void main()
{
    vec2 pos = aPos * size + offset;
    gl_Position = projection * vec4(pos, 0.0, 1.0);
    TexCoord = aUV;
}
)";
            default:
                throw std::runtime_error("Unknown shader type");
        }
    }

    static const char* fragmentSource(ShaderType type) {
        switch (type) {
            case ShaderType::GRID:
    return R"(
#version 330 core
out vec4 FragColor;

uniform vec4 uColor;

void main()
{
    FragColor = uColor;
}
)";
            case ShaderType::BLINN_PHONG:
                return R"(
#version 330 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 lightPos2;
uniform vec3 lightPos3;
uniform vec3 viewPos;

vec3 calculateLight(vec3 lightPosition, vec3 norm, vec3 viewDir, vec3 baseColor, float intensity)
{
    vec3 lightDir = normalize(lightPosition - FragPos);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * baseColor * intensity;

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 32.0);
    vec3 specular = 0.35 * spec * vec3(1.0) * intensity;

    return diffuse + specular;
}

void main()
{
    vec3 baseColor = vec3(0.88, 0.94, 0.98);
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 ambient = 0.15 * baseColor;

    vec3 light1 = calculateLight(lightPos, norm, viewDir, baseColor, 1.0);

    vec3 light2 = calculateLight(lightPos2, norm, viewDir, baseColor, 0.5);

    vec3 light3 = calculateLight(lightPos3, norm, viewDir, baseColor, 0.7);

    vec3 color = ambient + light1 + light2 + light3;

    FragColor = vec4(color, 1.0);
}
)";
            case ShaderType::SHADOW_PASS:
                return R"(
        #version 330 core
        void main()
        {
        }
    )";
            case ShaderType::WIREFRAME:
                return R"(
        #version 330 core
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    )";
            case ShaderType::UI:
return R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D fontTexture;
uniform vec4 color;

void main()
{
    float alpha = texture(fontTexture, TexCoord).r;
    FragColor = vec4(color.rgb, alpha * color.a);
}
)";
            default:
                throw std::runtime_error("Unknown shader type");
        }
    }
};

class DefaultShaderProvider : public IShaderProvider {
public:
    GLuint getShader(ShaderType type) override {
        return ShaderFactory::getShader(type);
    }
};

static void appendTriangle(
    std::vector<float>& data,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    const glm::vec3& normal
) {
    auto pushVertex = [&](const glm::vec3& v) {
        data.push_back(v.x);
        data.push_back(v.y);
        data.push_back(v.z);
        data.push_back(normal.x);
        data.push_back(normal.y);
        data.push_back(normal.z);
    };

    pushVertex(v0);
    pushVertex(v1);
    pushVertex(v2);
}

struct RenderParams {
    GLuint shader;
    glm::mat4 model;
    glm::mat4 mvp;
    glm::mat4 lightSpaceMatrix;
    glm::vec3 lightPos;
    glm::vec3 viewPos;
    bool isShadowPass;
};

class IPlatonicSolid {
public:
    virtual ~IPlatonicSolid() = default;

    virtual void render(const RenderParams& params) = 0;
};

class IObserver {
public:
    virtual ~IObserver() = default;

    virtual void onActiveObjectChanged(int newIndex) {}

    virtual void onCameraChanged(const glm::vec3& newPosition) {}

    virtual const void* getObserverId() const {
        return this;
    }
};

class ConsoleObserver : public IObserver {
public:
    void onActiveObjectChanged(int newIndex) override {
        std::cout << "Active object changed to index: "
                  << newIndex << std::endl;
    }

    void onCameraChanged(const glm::vec3& newPosition) override {
        std::cout
            << "Camera position changed: ("
            << newPosition.x << ", "
            << newPosition.y << ", "
            << newPosition.z << ")"
            << std::endl;
    }
};

class PlatonicSolid : public IPlatonicSolid {
public:
    virtual ~PlatonicSolid() = default;

    void render(const RenderParams& params) override
    {
        if (!mesh) return;

        glUseProgram(params.shader);

        const ShaderUniforms& u =
            ShaderManager::getInstance().getUniforms(params.shader);

        glUniformMatrix4fv(u.mvp, 1, GL_FALSE, glm::value_ptr(params.mvp));

        if (!params.isShadowPass) {
            glUniformMatrix4fv(u.model, 1, GL_FALSE, glm::value_ptr(params.model));

            glm::mat3 normalMatrix =
                glm::transpose(glm::inverse(glm::mat3(params.model)));

            glUniformMatrix3fv(u.normal, 1, GL_FALSE, glm::value_ptr(normalMatrix));

            glUniform3fv(u.light, 1, glm::value_ptr(params.lightPos));
            glm::vec3 lightPos2 = glm::vec3(-2.0f, 1.0f, 2.0f);
            glm::vec3 lightPos3 = glm::vec3(0.0f, 3.0f, -2.0f);

            if (u.lightPos2 != -1)
                glUniform3fv(u.lightPos2, 1, glm::value_ptr(lightPos2));

            if (u.lightPos3 != -1)
                glUniform3fv(u.lightPos3, 1, glm::value_ptr(lightPos3));
            glUniform3fv(u.view, 1, glm::value_ptr(params.viewPos));

            glUniformMatrix4fv(
                u.lightSpaceMatrix,
                1,
                GL_FALSE,
                glm::value_ptr(params.lightSpaceMatrix)
            );

            glUniform1i(u.shadowMap, 0);
        }

        mesh->draw();
    }

    int getVertexCount() const { return verticesCount; }
    int getFaceCount() const { return facesCount; }
    int getEdgeCount() const { return edgesCount; }

    void scaleToSize(std::vector<Vertex>& vertices, float target)
    {
        float maxLen = 0.0f;

        for (const auto& v : vertices) {
            float len = glm::length(v.position);
            if (len > maxLen) maxLen = len;
        }

        float scale = target / maxLen;

        for (auto& v : vertices) {
            v.position *= scale;
        }
    }

    void alignToGround(std::vector<Vertex>& vertices, float groundY)
    {
        float minY = std::numeric_limits<float>::max();

        for (const auto& v : vertices) {
            if (v.position.y < minY) {
                minY = v.position.y;
            }
        }

        float offset = groundY - minY;

        for (auto& v : vertices) {
            v.position.y += offset;
        }
    }

protected:
    PlatonicSolid() = default;

    int verticesCount = 0;
    int facesCount = 0;
    int edgesCount = 0;

    void buildMesh(const std::vector<Vertex>& v) {
        mesh = std::make_unique<Mesh>(v);
    }

    void calculateNormals(std::vector<Vertex>& vertices) {
        for (size_t i = 0; i < vertices.size(); i += 3) {
            glm::vec3& v0 = vertices[i].position;
            glm::vec3& v1 = vertices[i+1].position;
            glm::vec3& v2 = vertices[i+2].position;

            glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

            vertices[i].normal = normal;
            vertices[i+1].normal = normal;
            vertices[i+2].normal = normal;
        }
    }

    std::unique_ptr<Mesh> mesh;

    virtual void initializeGeometry() = 0;    
};

class WireframeDecorator : public IPlatonicSolid {
public:
    explicit WireframeDecorator(std::shared_ptr<IPlatonicSolid> component)
        : component(std::move(component))
    {
        if (!this->component) {
            throw std::runtime_error("WireframeDecorator: component is null");
        }
    }

    const std::shared_ptr<IPlatonicSolid>& getComponent() const {
        return component;
    }

    void render(const RenderParams& params) override
    {
        if (params.isShadowPass) {
            component->render(params);
            return;
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(2.5f);

        RenderParams wireParams = params;
        wireParams.shader = ShaderFactory::getShader(ShaderType::WIREFRAME);

        component->render(wireParams);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

private:
    std::shared_ptr<IPlatonicSolid> component;
};

class Cube : public PlatonicSolid {
public:
    Cube() : PlatonicSolid() {
        initializeGeometry();
    }

    void initializeGeometry() override {
        MeshBuilder mb;

        std::array<glm::vec3, 8> v = {
            glm::vec3(-0.5f,-0.5f,-0.5f),
            glm::vec3( 0.5f,-0.5f,-0.5f),
            glm::vec3( 0.5f, 0.5f,-0.5f),
            glm::vec3(-0.5f, 0.5f,-0.5f),
            glm::vec3(-0.5f,-0.5f, 0.5f),
            glm::vec3( 0.5f,-0.5f, 0.5f),
            glm::vec3( 0.5f, 0.5f, 0.5f),
            glm::vec3(-0.5f, 0.5f, 0.5f)
        };

        std::array<std::array<int,3>,12> f = {{
            {{0,1,2}},{{2,3,0}},
            {{4,6,5}},{{6,4,7}},
            {{7,0,3}},{{0,7,4}},
            {{6,2,1}},{{1,5,6}},
            {{0,4,5}},{{5,1,0}},
            {{3,2,6}},{{6,7,3}}
        }};

        for (auto& tri : f) {
            mb.addTriangle(
                v[tri[0]],
                v[tri[1]],
                v[tri[2]]
            );
        }

        calculateNormals(mb.vertices);
        buildMesh(mb.vertices);
        verticesCount = 8;
        facesCount = 6;
        edgesCount = 12;
    }
};


class Tetrahedron : public PlatonicSolid {
public:
    Tetrahedron() : PlatonicSolid() {
        initializeGeometry();
    }

    void initializeGeometry() override {
        MeshBuilder mb;

    const float s = 0.5f;

    std::array<glm::vec3, 4> v = {
        glm::vec3(0.0f,  std::sqrt(2.0f/3.0f), 0.0f),
        glm::vec3(-0.5f, 0.0f,  std::sqrt(3.0f)/6.0f),
        glm::vec3( 0.5f, 0.0f,  std::sqrt(3.0f)/6.0f),
        glm::vec3( 0.0f, 0.0f, -std::sqrt(3.0f)/3.0f)
    };

        std::array<std::array<int,3>,4> f = {{
            {{0,1,2}},
            {{0,3,1}},
            {{0,2,3}},
            {{1,3,2}}
        }};

        for (auto& tri : f) {
            mb.addTriangle(v[tri[0]], v[tri[1]], v[tri[2]]);
        }

        calculateNormals(mb.vertices);
        alignToGround(mb.vertices, -0.5f);
        buildMesh(mb.vertices);

        verticesCount = 4;
        facesCount = 4;
        edgesCount = 6;
    }
};

class Octahedron : public PlatonicSolid {
public:
    Octahedron() : PlatonicSolid() {
        initializeGeometry();
    }

    void initializeGeometry() override {
        MeshBuilder mb;

        float s = 0.5f;

        std::array<glm::vec3, 6> v = {
            glm::vec3( s, 0, 0),
            glm::vec3(-s, 0, 0),
            glm::vec3(0,  s, 0),
            glm::vec3(0, -s, 0),
            glm::vec3(0, 0,  s),
            glm::vec3(0, 0, -s)
        };

        std::array<std::array<int,3>,8> f = {{
            {{2,0,4}}, {{2,4,1}}, {{2,1,5}}, {{2,5,0}},
            {{3,4,0}}, {{3,1,4}}, {{3,5,1}}, {{3,0,5}}
        }};

        for (auto& tri : f) {
            mb.addTriangle(v[tri[0]], v[tri[1]], v[tri[2]]);
        }

        calculateNormals(mb.vertices);
        buildMesh(mb.vertices);

        verticesCount = 6;
        facesCount = 8;
        edgesCount = 12;
    }
};


class Icosahedron : public PlatonicSolid {
public:
    Icosahedron() : PlatonicSolid() {
        initializeGeometry();
    }

    void initializeGeometry() override {
        MeshBuilder mb;

        const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
        const float s = 0.35f;

        std::array<glm::vec3, 12> v = {
            glm::vec3(-1,  phi, 0)*s, glm::vec3( 1,  phi, 0)*s,
            glm::vec3(-1, -phi, 0)*s, glm::vec3( 1, -phi, 0)*s,
            glm::vec3(0, -1,  phi)*s, glm::vec3(0,  1,  phi)*s,
            glm::vec3(0, -1, -phi)*s, glm::vec3(0,  1, -phi)*s,
            glm::vec3( phi,0,-1)*s,   glm::vec3( phi,0, 1)*s,
            glm::vec3(-phi,0,-1)*s,   glm::vec3(-phi,0, 1)*s
        };

        std::array<std::array<int,3>,20> f = {{
            {{0,11,5}}, {{0,5,1}}, {{0,1,7}}, {{0,7,10}}, {{0,10,11}},
            {{1,5,9}}, {{5,11,4}}, {{11,10,2}}, {{10,7,6}}, {{7,1,8}},
            {{3,9,4}}, {{3,4,2}}, {{3,2,6}}, {{3,6,8}}, {{3,8,9}},
            {{4,9,5}}, {{2,4,11}}, {{6,2,10}}, {{8,6,7}}, {{9,8,1}}
        }};

        for (auto& tri : f) {
            mb.addTriangle(v[tri[0]], v[tri[1]], v[tri[2]]);
        }

        calculateNormals(mb.vertices);
        scaleToSize(mb.vertices, 0.5f);
        alignToGround(mb.vertices, -0.5f);
        buildMesh(mb.vertices);

        verticesCount = 12;
        facesCount = 20;
        edgesCount = 30;
    }
};

class Dodecahedron : public PlatonicSolid {
public:
    Dodecahedron() : PlatonicSolid() {
        initializeGeometry();
    }

    void initializeGeometry() override {
        MeshBuilder mb;

        const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
        const float s = 0.35f;

        std::array<glm::vec3, 12> iv = {
            glm::vec3(-1,  phi, 0)*s, glm::vec3( 1,  phi, 0)*s,
            glm::vec3(-1, -phi, 0)*s, glm::vec3( 1, -phi, 0)*s,
            glm::vec3(0, -1,  phi)*s, glm::vec3(0,  1,  phi)*s,
            glm::vec3(0, -1, -phi)*s, glm::vec3(0,  1, -phi)*s,
            glm::vec3( phi,0,-1)*s,   glm::vec3( phi,0, 1)*s,
            glm::vec3(-phi,0,-1)*s,   glm::vec3(-phi,0, 1)*s
        };

        std::array<std::array<int,3>,20> faces = {{
            {{0,11,5}}, {{0,5,1}}, {{0,1,7}}, {{0,7,10}}, {{0,10,11}},
            {{1,5,9}}, {{5,11,4}}, {{11,10,2}}, {{10,7,6}}, {{7,1,8}},
            {{3,9,4}}, {{3,4,2}}, {{3,2,6}}, {{3,6,8}}, {{3,8,9}},
            {{4,9,5}}, {{2,4,11}}, {{6,2,10}}, {{8,6,7}}, {{9,8,1}}
        }};

        std::array<glm::vec3, 20> centers;
        for (size_t i = 0; i < faces.size(); ++i) {
            centers[i] = glm::normalize(
                (iv[faces[i][0]] +
                 iv[faces[i][1]] +
                 iv[faces[i][2]]) / 3.0f
            );
        }

        std::array<std::vector<int>, 12> around;
        for (int f = 0; f < 20; ++f) {
            for (int v = 0; v < 3; ++v) {
                around[faces[f][v]].push_back(f);
            }
        }

        for (int i = 0; i < 12; ++i) {
            auto& list = around[i];
            if (list.size() != 5) continue;

            const glm::vec3 center = iv[i];

            glm::vec3 normal(0.0f);
            for (int idx : list) {
                normal += centers[idx] - center;
            }
            normal = glm::normalize(normal);

            glm::vec3 tangent = glm::normalize(centers[list[0]] - center);
            glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

            std::sort(list.begin(), list.end(), [&](int a, int b) {
                glm::vec3 da = centers[a] - center;
                glm::vec3 db = centers[b] - center;

                float ax = glm::dot(da, tangent);
                float ay = glm::dot(da, bitangent);

                float bx = glm::dot(db, tangent);
                float by = glm::dot(db, bitangent);

                return std::atan2(ay, ax) < std::atan2(by, bx);
            });

            glm::vec3 faceCenter(0.0f);
            for (int idx : list) {
                faceCenter += centers[idx];
            }
            faceCenter /= 5.0f;

            for (int k = 0; k < 5; ++k) {
                const glm::vec3& v0 = centers[list[k]];
                const glm::vec3& v1 = centers[list[(k + 1) % 5]];

                mb.addTriangle(faceCenter, v0, v1);
            }
        }

        calculateNormals(mb.vertices);
        scaleToSize(mb.vertices, 0.5f);
        alignToGround(mb.vertices, -0.5f);
        buildMesh(mb.vertices);

        verticesCount = 20;
        facesCount = 12;
        edgesCount = 30;
    }
};

class Camera {
public:
    void addObserver(IObserver* observer) {
        if (!observer) return;

        auto it = std::find_if(observers.begin(), observers.end(),
            [&](IObserver* o) {
                return o && o->getObserverId() == observer->getObserverId();
            });

        if (it == observers.end()) {
            observers.push_back(observer);
        }
    }

    void removeObserver(IObserver* observer) {
        if (!observer) return;

        observers.erase(
            std::remove_if(observers.begin(), observers.end(),
                [&](IObserver* o) {
                    return o && o->getObserverId() == observer->getObserverId();
                }),
            observers.end()
        );
    }

    Camera(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
        : target(target), up(up) {
        const glm::vec3 offset = position - target;
        radius = glm::length(offset);

        if (radius > 1e-6f) {
            const float horizontal = std::sqrt(offset.x * offset.x + offset.z * offset.z);
            pitch = glm::degrees(std::atan2(offset.y, horizontal));
            yaw = glm::degrees(std::atan2(offset.x, offset.z));
            pitch = glm::clamp(pitch, -89.0f, 89.0f);
            normalizeYaw();
        } else {
            pitch = 0.0f;
            yaw = 0.0f;
            radius = 3.0f;
        }

    }

    glm::mat4 getViewMatrix() const {
        const glm::vec3 position = calculatePosition();
        return glm::lookAt(position, target, up);
    }

    glm::mat4 getProjectionMatrix(float aspect) const {
        return glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    }

    glm::vec3 getPosition() const {
        return calculatePosition();
    }

    void rotate(float yawDelta, float pitchDelta) {
        yaw += yawDelta;
        normalizeYaw();

        pitch += pitchDelta;
        pitch = glm::clamp(pitch, -89.0f, 89.0f);

        notifyObservers();
    }

    void zoom(float amount) {
        radius += amount;
        radius = glm::clamp(radius, 1.5f, 10.0f);

        notifyObservers();
    }

private:
    glm::vec3 calculatePosition() const {
        const float yawRad = glm::radians(yaw);
        const float pitchRad = glm::radians(pitch);
        return glm::vec3(
            target.x + radius * std::cos(pitchRad) * std::sin(yawRad),
            target.y + radius * std::sin(pitchRad),
            target.z + radius * std::cos(pitchRad) * std::cos(yawRad)
        );
    }

    void normalizeYaw() {
        yaw = std::fmod(yaw, 360.0f);
        if (yaw > 180.0f) {
            yaw -= 360.0f;
        } else if (yaw < -180.0f) {
            yaw += 360.0f;
        }
    }

    void notifyObservers() {
        const glm::vec3 position = getPosition();

        for (IObserver* observer : observers) {
            if (!observer) continue;
            observer->onCameraChanged(position);
        }
    }

    std::vector<IObserver*> observers;

    glm::vec3 target;
    glm::vec3 up;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float radius = 3.0f;
};

class SolidManager {
public:
    void addObject(const std::shared_ptr<IPlatonicSolid>& object) {
        objects.push_back(object);
    }

    void setActiveIndex(int index) {
        if (index >= 0 &&
            index < static_cast<int>(objects.size()) &&
            index != activeIndex)
        {
            activeIndex = index;
            notifyObservers();
        }
    }

    void addObserver(IObserver* observer) {
        if (!observer) return;

        auto it = std::find_if(observers.begin(), observers.end(),
            [&](IObserver* o) {
                return o && o->getObserverId() == observer->getObserverId();
            });

        if (it == observers.end()) {
            observers.push_back(observer);
        }
    }

    void removeObserver(IObserver* observer) {
        if (!observer) return;

        observers.erase(
            std::remove_if(observers.begin(), observers.end(),
                [&](IObserver* o) {
                    return o && o->getObserverId() == observer->getObserverId();
                }),
            observers.end()
        );
    }

    int getActiveIndex() const {
        return activeIndex;
    }

    std::shared_ptr<IPlatonicSolid> getActiveObject() const {
        if (activeIndex < 0 || activeIndex >= static_cast<int>(objects.size())) {
            return nullptr;
        }
        return objects[static_cast<size_t>(activeIndex)];
    }

    std::vector<std::shared_ptr<IPlatonicSolid>>& getObjectsMutable() {
        return objects;
    }

    const std::vector<std::shared_ptr<IPlatonicSolid>>& getObjects() const {
        return objects;
    }

private:
    void notifyObservers() {
        for (IObserver* observer : observers) {
            if (!observer) continue;
            observer->onActiveObjectChanged(activeIndex);
        }
    }

    std::vector<std::shared_ptr<IPlatonicSolid>> objects;
    std::vector<IObserver*> observers;
    int activeIndex = 0;
};

class IScene {
public:
    virtual ~IScene() = default;

    virtual void render(GLuint shader,
                    const glm::mat4& model,
                    const glm::mat4& mvp,
                    const glm::mat4& lightSpaceMatrix,
                    const glm::vec3& lightPos,
                    const glm::vec3& viewPos,
                    bool isShadowPass) = 0;
};

class Scene : public IScene {
public:
    explicit Scene(SolidManager* manager) : manager(manager) {
    }

    void render(GLuint shader,
            const glm::mat4& model,
            const glm::mat4& mvp,
            const glm::mat4& lightSpaceMatrix,
            const glm::vec3& lightPos,
            const glm::vec3& viewPos,
            bool isShadowPass) override {
        if (!manager) {
            return;
        }
        const std::shared_ptr<IPlatonicSolid> activeObject = manager->getActiveObject();
        if (activeObject) {
            RenderParams params;
            params.shader = shader;
            params.model = model;
            params.mvp = mvp;
            params.lightPos = lightPos;
            params.viewPos = viewPos;
            params.isShadowPass = isShadowPass;

            params.lightSpaceMatrix = lightSpaceMatrix;

            activeObject->render(params);
        }
    }

private:
    SolidManager* manager = nullptr;
};

struct InputState {
    bool leftMousePressed = false;
    float mouseDeltaX = 0.0f;
    float mouseDeltaY = 0.0f;
    float scrollDelta = 0.0f;
    bool leftKeyPressed = false;
    bool rightKeyPressed = false;
    bool upKeyPressed = false;
    bool downKeyPressed = false;
    bool numberKeyPressed[5] = {false, false, false, false, false};
    bool toggleWireframe = false;
    bool toggleFPS = false;
    double mouseX = 0.0;
    double mouseY = 0.0;
    bool mouseClicked = false;
};

struct UIButton {
    float x, y;
    float width, height;
};

struct Glyph
{
    int id;

    float x;
    float y;
    float width;
    float height;

    float xoffset;
    float yoffset;
    float xadvance;

    float u0;
    float v0;
    float u1;
    float v1;
};

class WindowSystem {
public:
    WindowSystem() {
        if (!glfwInit()) {
            throw std::runtime_error("GLFW init failed");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Platon Engine", nullptr, nullptr);

        glfwMakeContextCurrent(window);

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        glfwSetWindowSize(window, mode->width, mode->height);
        glfwSetWindowPos(window, 0, 0);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            glfwDestroyWindow(window);
            window = nullptr;
            glfwTerminate();
            throw std::runtime_error("Failed to init GLAD");
        }

        refreshFramebufferSize();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CW);
    }

    ~WindowSystem() {
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        glfwTerminate();
    }

    GLFWwindow* getWindow() const {
        return window;
    }

    void swapBuffers() const {
        glfwSwapBuffers(window);
    }

    void pollEvents() const {
        glfwPollEvents();
    }

    bool shouldClose() const {
        return glfwWindowShouldClose(window) != 0;
    }

    void refreshFramebufferSize() {
        int width = WIDTH;
        int height = HEIGHT;
        glfwGetFramebufferSize(window, &width, &height);
        width = std::max(width, 1);
        height = std::max(height, 1);
        if (width != framebufferWidth || height != framebufferHeight) {
            framebufferWidth = width;
            framebufferHeight = height;
            glViewport(0, 0, framebufferWidth, framebufferHeight);
        }
    }

    float getAspectRatio() const {
        return static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight);
    }

private:
    GLFWwindow* window = nullptr;
    int framebufferWidth = WIDTH;
    int framebufferHeight = HEIGHT;
};

class InputSystem {
public:
    InputState& getState() {
        return inputState;
    }

    void onMouseButton(int button, int action) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            inputState.leftMousePressed = (action == GLFW_PRESS);
            if (action == GLFW_PRESS || action == GLFW_RELEASE) {
                hasLastMousePosition = false;
            }
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
                inputState.mouseClicked = true;
            }
        }
    }

    void onCursorPosition(double xpos, double ypos) {
        inputState.mouseX = xpos;
        inputState.mouseY = ypos;
        if (!inputState.leftMousePressed) {
            return;
        }

        if (!hasLastMousePosition) {
            lastMouseX = xpos;
            lastMouseY = ypos;
            hasLastMousePosition = true;
            return;
        }

        const float deltaX = static_cast<float>(xpos - lastMouseX);
        const float deltaY = static_cast<float>(ypos - lastMouseY);
        lastMouseX = xpos;
        lastMouseY = ypos;

        inputState.mouseDeltaX += deltaX;
        inputState.mouseDeltaY += deltaY;
    }

    void onScroll(double yoffset) {
        inputState.scrollDelta += static_cast<float>(yoffset);
    }

    void onKey(int key, int action) {
        if (action != GLFW_PRESS && action != GLFW_RELEASE && action != GLFW_REPEAT) {
            return;
        }

        const bool isPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
        if (key == GLFW_KEY_LEFT) {
            inputState.leftKeyPressed = isPressed;
        } else if (key == GLFW_KEY_RIGHT) {
            inputState.rightKeyPressed = isPressed;
        } else if (key == GLFW_KEY_UP) {
            inputState.upKeyPressed = isPressed;
        } else if (key == GLFW_KEY_DOWN) {
            inputState.downKeyPressed = isPressed;
        } else if (key >= GLFW_KEY_1 && key <= GLFW_KEY_5 && isPressed) {
            inputState.numberKeyPressed[key - GLFW_KEY_1] = true;
        } else if (key == GLFW_KEY_W && action == GLFW_PRESS) {
            inputState.toggleWireframe = true;
        } else if (key == GLFW_KEY_F && action == GLFW_PRESS) {
            inputState.toggleFPS = true;
        }
    }

private:
    InputState inputState;
    bool hasLastMousePosition = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
};

struct RenderContext {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 cameraPos;
    glm::vec3 lightPos;
    int activeIndex;
    int hoveredIndex;
    int fps;
    bool showFPS;
    bool showInfoWindow = false;

    int verticesCount = 0;
    int facesCount = 0;
    int edgesCount = 0;
};

class GraphicsSystem {
public:
    GraphicsSystem(std::shared_ptr<IShaderProvider> provider)
    : shaderProvider(provider)
    {
        initGrid();
        gridShader = shaderProvider->getShader(ShaderType::GRID);

        float quad[] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,

            1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f
        };

        glGenVertexArrays(1, &uiVAO);
        glGenBuffers(1, &uiVBO);

        glBindVertexArray(uiVAO);
        glBindBuffer(GL_ARRAY_BUFFER, uiVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(0);

        loadBitmapFont("font.fnt");

        glGenTextures(1, &fontTexture);
        glBindTexture(GL_TEXTURE_2D, fontTexture);

        int width, height, channels;
        stbi_set_flip_vertically_on_load(false);

        unsigned char* data = stbi_load(
            "font.png",
            &width,
            &height,
            &channels,
            1
        );

        if (!data)
        {
            throw std::runtime_error("Failed to load font.png");
        }

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            width,
            height,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            data
        );

        stbi_image_free(data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    void drawGlyph(
    char c,
    float x,
    float y,
    float size,
    const glm::vec4& textColor
    )
    {
        if (glyphs.find(c) == glyphs.end())
            return;

        const Glyph& g = glyphs[c];

        float xpos = x + g.xoffset;
        float ypos = y;

        float w = g.width;
        float h = g.height;

        float vertices[] = {
            0, 0, g.u0, g.v1,
            1, 0, g.u1, g.v1,
            0, 1, g.u0, g.v0,

            1, 0, g.u1, g.v1,
            1, 1, g.u1, g.v0,
            0, 1, g.u0, g.v0
        };

        glBindBuffer(GL_ARRAY_BUFFER, uiVBO);
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            sizeof(vertices),
            vertices
        );

        GLuint uiShader = shaderProvider->getShader(ShaderType::UI);
        glUseProgram(uiShader);

        int screenW, screenH;
        glfwGetFramebufferSize(
            windowSystem.getWindow(),
            &screenW,
            &screenH
        );

        glm::mat4 proj = glm::ortho(
            0.0f,
            (float)screenW,
            0.0f,
            (float)screenH
        );

        glUniformMatrix4fv(
            glGetUniformLocation(uiShader, "projection"),
            1,
            GL_FALSE,
            glm::value_ptr(proj)
        );

        glUniform2f(
            glGetUniformLocation(uiShader, "offset"),
            xpos,
            ypos
        );

        glUniform2f(
            glGetUniformLocation(uiShader, "size"),
            w,
            h
        );

        glUniform4f(
            glGetUniformLocation(uiShader, "color"),
            textColor.r,
            textColor.g,
            textColor.b,
            textColor.a
        );

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fontTexture);

        glUniform1i(
            glGetUniformLocation(uiShader, "fontTexture"),
            0
        );

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glBindVertexArray(uiVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    float getTextWidth(const std::string& text)
    {
        float width = 0.0f;

        for (char c : text)
        {
            if (glyphs.find(c) == glyphs.end())
                continue;

            width += glyphs[c].xadvance;
        }

        return width;
    }

    void drawText(
        const std::string& text,
        float x,
        float y,
        const glm::vec4& textColor
    )
    {
        for (char c : text)
        {
            if (glyphs.find(c) == glyphs.end())
                continue;

            const Glyph& g = glyphs[c];

            drawGlyph(
                c,
                x,
                y,
                g.height,
                textColor
            );

            x += g.xadvance;
        }
    }

    std::vector<std::string> wrapText(const std::string& text, float maxWidth)
    {
        std::vector<std::string> lines;
        std::stringstream ss(text);
        std::string word;
        std::string currentLine;

        while (ss >> word)
        {
            std::string testLine = currentLine.empty()
                ? word
                : currentLine + " " + word;

            if (getTextWidth(testLine) > maxWidth)
            {
                if (!currentLine.empty())
                    lines.push_back(currentLine);

                currentLine = word;
            }
            else
            {
                currentLine = testLine;
            }
        }

        if (!currentLine.empty())
            lines.push_back(currentLine);

        return lines;
    }

    void render(IScene& scene, const RenderContext& ctx);

    void renderGrid(const RenderContext& ctx)
    {
        glLineWidth(1.0f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(gridShader);

        glm::mat4 model(1.0f);
        glm::mat4 mvp = ctx.projection * ctx.view * model;

        GLint mvpLoc = glGetUniformLocation(gridShader, "uMVP");
        GLint colorLoc = glGetUniformLocation(gridShader, "uColor");

        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        glUniform4f(colorLoc, 0.5f, 0.5f, 0.5f, 0.7f);

        glBindVertexArray(gridVAO);
        glDrawArrays(GL_LINES, 0, gridVertexCount);
        glBindVertexArray(0);

        glDisable(GL_BLEND);
    }

    ~GraphicsSystem() {
        if (depthMap != 0) {
            glDeleteTextures(1, &depthMap);
            depthMap = 0;
        }
        if (shadowFBO != 0) {
            glDeleteFramebuffers(1, &shadowFBO);
            shadowFBO = 0;
        }
        if (gridVBO) glDeleteBuffers(1, &gridVBO);
        if (gridVAO) glDeleteVertexArrays(1, &gridVAO);
    }

    void renderShadowPass(IScene& scene,
                      const glm::mat4& model,
                      const glm::mat4& lightSpaceMatrix,
                      const glm::vec3& lightPos,
                      const glm::vec3& viewPos);

    void renderLightingPass(IScene& scene,
                    const glm::mat4& model,
                    const glm::mat4& view,
                    const glm::mat4& projection,
                    const glm::mat4& lightSpaceMatrix,
                    const glm::vec3& lightPos,
                    const glm::vec3& viewPos);

    GLFWwindow* getWindow() const {
        return windowSystem.getWindow();
    }

    bool shouldClose() const {
        return windowSystem.shouldClose();
    }

    float getAspectRatio() const {
        return windowSystem.getAspectRatio();
    }

    float beginFrame() {
        const float currentTime = static_cast<float>(glfwGetTime());
        const float deltaTime = currentTime - previousTime;
        previousTime = currentTime;
        return deltaTime;
    }

    void prepareFrame() {
        windowSystem.refreshFramebufferSize();
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void finalizeFrame() {
        windowSystem.swapBuffers();
        windowSystem.pollEvents();
    }

    void setupShadowMap(unsigned int width, unsigned int height) {
        if (shadowFBO != 0) {
            glDeleteFramebuffers(1, &shadowFBO);
            shadowFBO = 0;
        }
        if (depthMap != 0) {
            glDeleteTextures(1, &depthMap);
            depthMap = 0;
        }

        shadowMapWidth = width;
        shadowMapHeight = height;

        glGenFramebuffers(1, &shadowFBO);
        if (shadowFBO == 0) {
            throw std::runtime_error("Failed to create shadow framebuffer");
        }

        glGenTextures(1, &depthMap);
        if (depthMap == 0) {
            glDeleteFramebuffers(1, &shadowFBO);
            shadowFBO = 0;
            throw std::runtime_error("Failed to create shadow depth texture");
        }

        glBindTexture(GL_TEXTURE_2D, depthMap);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const float borderColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
        glCullFace(GL_FRONT);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glCullFace(GL_BACK);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteTextures(1, &depthMap);
            glDeleteFramebuffers(1, &shadowFBO);
            depthMap = 0;
            shadowFBO = 0;
            throw std::runtime_error("Shadow framebuffer is incomplete");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

private:
    WindowSystem windowSystem;
    float previousTime = static_cast<float>(glfwGetTime());
    GLuint shadowFBO = 0;
    GLuint depthMap = 0;
    unsigned int shadowMapWidth = 0;
    unsigned int shadowMapHeight = 0;
    std::shared_ptr<IShaderProvider> shaderProvider;

    GLuint gridVAO = 0;
    GLuint gridVBO = 0;
    GLsizei gridVertexCount = 0;
    GLuint gridShader = 0;

    void initGrid()
    {
        std::vector<glm::vec3> vertices;

        float size = 5.0f;
        float step = 0.5f;

        for (float i = -size; i <= size; i += step)
        {
           float y = -0.5f;

        vertices.push_back(glm::vec3(-size, y, i));
        vertices.push_back(glm::vec3(size, y, i));

        vertices.push_back(glm::vec3(i, y, -size));
        vertices.push_back(glm::vec3(i, y, size));
        }

        gridVertexCount = static_cast<GLsizei>(vertices.size());

        glGenVertexArrays(1, &gridVAO);
        glGenBuffers(1, &gridVBO);

        glBindVertexArray(gridVAO);
        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);

        glBufferData(
            GL_ARRAY_BUFFER,
            vertices.size() * sizeof(glm::vec3),
            vertices.data(),
            GL_STATIC_DRAW
        );

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                            sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);
    }

    void loadBitmapFont(const std::string& fntPath)
    {
        std::ifstream file(fntPath);

        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open font file: " + fntPath);
        }

        std::string line;

        while (std::getline(file, line))
        {
            if (line.find("scaleW=") != std::string::npos)
            {
                std::stringstream ss(line);
                std::string token;

                while (ss >> token)
                {
                    if (token.find("scaleW=") != std::string::npos)
                        fontTextureWidth = std::stoi(token.substr(7));

                    if (token.find("scaleH=") != std::string::npos)
                        fontTextureHeight = std::stoi(token.substr(7));
                }
            }

            if (line.find("char id=") != std::string::npos)
            {
                Glyph g{};
                std::stringstream ss(line);
                std::string token;

                while (ss >> token)
                {
                    if (token.find("id=") != std::string::npos)
                        g.id = std::stoi(token.substr(3));

                    else if (token.find("x=") != std::string::npos)
                        g.x = std::stof(token.substr(2));

                    else if (token.find("y=") != std::string::npos)
                        g.y = std::stof(token.substr(2));

                    else if (token.find("width=") != std::string::npos)
                        g.width = std::stof(token.substr(6));

                    else if (token.find("height=") != std::string::npos)
                        g.height = std::stof(token.substr(7));

                    else if (token.find("xoffset=") != std::string::npos)
                        g.xoffset = std::stof(token.substr(8));

                    else if (token.find("yoffset=") != std::string::npos)
                        g.yoffset = std::stof(token.substr(8));

                    else if (token.find("xadvance=") != std::string::npos)
                        g.xadvance = std::stof(token.substr(9));
                }

                g.u0 = g.x / fontTextureWidth;
                g.v0 = g.y / fontTextureHeight;
                g.u1 = (g.x + g.width) / fontTextureWidth;
                g.v1 = (g.y + g.height) / fontTextureHeight;

                glyphs[(char)g.id] = g;
            }
        }
    }

    GLuint uiVAO = 0;
    GLuint uiVBO = 0;

    GLuint textVAO = 0;
    GLuint textVBO = 0;

    GLuint fontTexture = 0;

    std::map<char, Glyph> glyphs;

    int fontTextureWidth = 0;
    int fontTextureHeight = 0;
};

void GraphicsSystem::render(IScene& scene, const RenderContext& ctx)
{
    glm::mat4 model(1.0f);

    glm::mat4 view = ctx.view;
    glm::mat4 projection = ctx.projection;

    glm::vec3 lightPos = ctx.lightPos;
    glm::vec3 viewPos = ctx.cameraPos;

    glm::mat4 lightView = glm::lookAt(
        lightPos,
        glm::vec3(0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );

    glm::mat4 lightProjection = glm::ortho(
        -5.0f, 5.0f,
        -5.0f, 5.0f,
        1.0f, 10.0f
    );

    glm::mat4 lightSpaceMatrix = lightProjection * lightView;

    renderShadowPass(scene, model, lightSpaceMatrix, lightPos, viewPos);

    renderLightingPass(scene, model, view, projection, lightSpaceMatrix, lightPos, viewPos);
    renderGrid(ctx);

    int activeIndex = ctx.activeIndex;
    std::string solidNames[5] =
    {
        "CUBE",
        "TETRAHEDRON",
        "OCTAHEDRON",
        "ICOSAHEDRON",
        "DODECAHEDRON"
    };

    std::string solidDescriptions[5] =
    {
        "THE CUBE SYMBOLIZES STABILITY AND ORDER. IN ANCIENT PHILOSOPHY IT WAS ASSOCIATED WITH THE ELEMENT OF EARTH, REPRESENTING STRUCTURE, BALANCE AND THE MATERIAL WORLD.",

        "THE TETRAHEDRON IS ONE OF THE SIMPLEST FORMS AND WAS LINKED TO FIRE BY PLATO. ITS SHARP AND DYNAMIC SHAPE REPRESENTS ENERGY, TRANSFORMATION AND CONSTANT MOTION.",

        "THE OCTAHEDRON WAS CONNECTED WITH AIR. ITS BALANCED SYMMETRY REFLECTS HARMONY AND LIGHTNESS, OFTEN SEEN AS A SYMBOL OF EQUILIBRIUM BETWEEN OPPOSING FORCES.",

        "THE ICOSAHEDRON WAS ASSOCIATED WITH WATER. ITS COMPLEX AND FLOWING FORM REPRESENTS CHANGE, ADAPTABILITY AND THE IDEA OF CONTINUOUS MOVEMENT.",

        "THE DODECAHEDRON HELD A SPECIAL PLACE IN PHILOSOPHY AND WAS LINKED TO THE COSMOS ITSELF. IT SYMBOLIZES THE UNIVERSE, MYSTERY AND THE IDEA OF A GREATER WHOLE."
    };

    GLuint uiShader = shaderProvider->getShader(ShaderType::UI);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(uiShader);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    int w, h;
    glfwGetFramebufferSize(windowSystem.getWindow(), &w, &h);

    glm::mat4 proj = glm::ortho(0.0f, (float)w, 0.0f, (float)h);
    glUniformMatrix4fv(glGetUniformLocation(uiShader, "projection"),
                    1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(uiVAO);

    float panelWidth = w * 0.12f;

    glUniform2f(glGetUniformLocation(uiShader, "offset"), w - panelWidth, 0);
    glUniform2f(glGetUniformLocation(uiShader, "size"), panelWidth, (float)h);
    glUniform4f(glGetUniformLocation(uiShader, "color"), 0.05f, 0.05f, 0.05f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    if (ctx.showInfoWindow)
    {
        float windowWidth = w * 0.3f;
        float windowHeight = h * 0.2f;

        float x = (w - windowWidth) * 0.5f;
        float y = (h - windowHeight) * 0.98f;

        glUniform2f(glGetUniformLocation(uiShader, "offset"), x, y);
        glUniform2f(glGetUniformLocation(uiShader, "size"), windowWidth, windowHeight);
        glUniform4f(glGetUniformLocation(uiShader, "color"), 0.1f, 0.1f, 0.1f, 0.95f);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        if (activeIndex >= 0 && activeIndex < 5)
        {
            std::string title = solidNames[activeIndex];

            float textWidth = getTextWidth(title);

            float textX = x + (windowWidth - textWidth) * 0.5f;
            float textY = y + windowHeight * 0.82f;

            drawText(
                title,
                textX,
                textY,
                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)
            );

            if (activeIndex >= 0 && activeIndex < 5)
            {
                std::string desc = solidDescriptions[activeIndex];

                float padding = 20.0f;
                float textX = x + padding;
                float textY = y + windowHeight * 0.6f;

                float maxTextWidth = windowWidth - padding * 2.0f;

                std::vector<std::string> lines =
                    wrapText(desc, maxTextWidth);

                float lineHeight = 22.0f;

                for (size_t i = 0; i < lines.size(); ++i)
                {
                    drawText(
                        lines[i],
                        textX,
                        textY - i * lineHeight,
                        glm::vec4(0.8f, 0.8f, 0.8f, 1.0f)
                    );
                }
            }
        }
    }

    float buttonSize = panelWidth * 0.5f;
    float spacing = panelWidth * 0.15f;

    for (int i = 0; i < 5; i++) {
        float startY = h - buttonSize - spacing;

        float y = startY - i * (buttonSize + spacing);
        float x = w - panelWidth + (panelWidth - buttonSize) / 2.0f;

        glUniform2f(glGetUniformLocation(uiShader, "offset"), x, y);
        glUniform2f(glGetUniformLocation(uiShader, "size"), buttonSize, buttonSize);
       if (i == activeIndex)
        {
            glUniform4f(glGetUniformLocation(uiShader, "color"),
                        0.8f, 0.5f, 0.2f, 1.0f);
        }
        else if (i == ctx.hoveredIndex)
        {
            glUniform4f(glGetUniformLocation(uiShader, "color"),
                        0.5f, 0.5f, 0.5f, 1.0f);
        }
        else
        {
            glUniform4f(glGetUniformLocation(uiShader, "color"),
                        0.25f, 0.25f, 0.25f, 1.0f);
        }

        std::string buttonTexts[5] =
        {
            "CUBE",
            "TETRA",
            "OCTA",
            "ICOSA",
            "DODECA"
        };

        glm::vec4 textColor;

        if (i == activeIndex)
        {
            textColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        }
        else if (i == ctx.hoveredIndex)
        {
            textColor = glm::vec4(0.95f, 0.95f, 0.95f, 1.0f);
        }
        else
        {
            textColor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        }

        float textWidth = getTextWidth(buttonTexts[i]);

        float textX = x + (buttonSize - textWidth) / 2.0f;
        float textY = y + buttonSize * 0.58f;

        drawText(
            buttonTexts[i],
            textX,
            textY,
            textColor
        );
    }

    float startY = h - buttonSize - spacing;
    float lastButtonY = startY - 4 * (buttonSize + spacing);

    bool infoHovered = (ctx.hoveredIndex == 5);
    bool viewHovered = (ctx.hoveredIndex == 6);

    std::string infoText = "INFO";

    float infoWidth = getTextWidth(infoText);
    float infoX = w - panelWidth + (panelWidth - infoWidth) / 2.0f;
    float infoY = lastButtonY - spacing * 2.0f;

    glm::vec4 infoColor;

    if (infoHovered)
        infoColor = glm::vec4(1.0f, 0.6f, 0.2f, 1.0f);
    else
        infoColor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);

    drawText(
        infoText,
        infoX,
        infoY,
        infoColor
    );

    std::string viewModeText = "VIEW MODE";

    float viewWidth = getTextWidth(viewModeText);
    float viewX = w - panelWidth + (panelWidth - viewWidth) / 2.0f;
    float viewY = infoY - spacing * 4.0f;

    glm::vec4 viewColor;

    if (viewHovered)
        viewColor = glm::vec4(0.3f, 0.8f, 1.0f, 1.0f);
    else
        viewColor = glm::vec4(0.6f, 0.6f, 1.0f, 1.0f);

    drawText(
        viewModeText,
        viewX,
        viewY,
        viewColor
    );

    glDisable(GL_BLEND);

    if (ctx.showFPS)
    {
        int w, h;
        glfwGetFramebufferSize(windowSystem.getWindow(), &w, &h);

        std::string fpsText = "FPS: " + std::to_string(ctx.fps);

        float textWidth = getTextWidth(fpsText);

        float x = w - textWidth - 15.0f;
        float y = 45.0f;

        drawText(
            fpsText,
            x,
            y,
            glm::vec4(1.0f, 1.0f, 0.3f, 1.0f)
        );
    }

    {
        float x = 15.0f;
        float y = h - 25.0f;

        float lineSpacing = 32.0f;

        drawText(
            "Vertices: " + std::to_string(ctx.verticesCount),
            x,
            y,
            glm::vec4(0.9f, 0.9f, 0.9f, 1.0f)
        );

        drawText(
            "Faces: " + std::to_string(ctx.facesCount),
            x,
            y - lineSpacing,
            glm::vec4(0.9f, 0.9f, 0.9f, 1.0f)
        );

        drawText(
            "Edges: " + std::to_string(ctx.edgesCount),
            x,
            y - 2.0f * lineSpacing,
            glm::vec4(0.9f, 0.9f, 0.9f, 1.0f)
        );
    }

    glEnable(GL_DEPTH_TEST);
}

void GraphicsSystem::renderShadowPass(
    IScene& scene,
    const glm::mat4& model,
    const glm::mat4& lightSpaceMatrix,
    const glm::vec3& lightPos,
    const glm::vec3& viewPos)
{
    glViewport(0, 0, shadowMapWidth, shadowMapHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glClear(GL_DEPTH_BUFFER_BIT);

    GLuint shadowShader = shaderProvider->getShader(ShaderType::SHADOW_PASS);
    glUseProgram(shadowShader);

    glm::mat4 lightMVP = lightSpaceMatrix * model;

    scene.render(
        shadowShader,
        model,
        lightMVP,
        lightSpaceMatrix,
        lightPos,
        viewPos,
        true
    );

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GraphicsSystem::renderLightingPass(
    IScene& scene,
    const glm::mat4& model,
    const glm::mat4& view,
    const glm::mat4& projection,
    const glm::mat4& lightSpaceMatrix,
    const glm::vec3& lightPos,
    const glm::vec3& viewPos)
{
    windowSystem.refreshFramebufferSize();

    int w = 800, h = 600;
    glfwGetFramebufferSize(windowSystem.getWindow(), &w, &h);
    glViewport(0, 0, w, h);

    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 mvp = projection * view * model;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthMap);

    GLuint lightingShader =
        shaderProvider->getShader(ShaderType::BLINN_PHONG);

    glUseProgram(lightingShader);

    scene.render(
        lightingShader,
        model,
        mvp,
        lightSpaceMatrix,
        lightPos,
        viewPos,
        false
    );
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    InputSystem* inputSystem = static_cast<InputSystem*>(glfwGetWindowUserPointer(window));
    if (!inputSystem) {
        return;
    }
    inputSystem->onMouseButton(button, action);
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    InputSystem* inputSystem = static_cast<InputSystem*>(glfwGetWindowUserPointer(window));
    if (!inputSystem) {
        return;
    }
    inputSystem->onCursorPosition(xpos, ypos);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)xoffset;
    InputSystem* inputSystem = static_cast<InputSystem*>(glfwGetWindowUserPointer(window));
    if (!inputSystem) {
        return;
    }
    inputSystem->onScroll(yoffset);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;
    InputSystem* inputSystem = static_cast<InputSystem*>(glfwGetWindowUserPointer(window));
    if (!inputSystem) {
        return;
    }

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }   
    inputSystem->onKey(key, action);
}

struct SolidInfo {
    int faces = 0;
    int vertices = 0;
    int edges = 0;
};

static std::array<SolidInfo, 5> solidStats = {
    SolidInfo{4, 4, 6},
    SolidInfo{6, 8, 12},
    SolidInfo{8, 6, 12},
    SolidInfo{20, 12, 30},
    SolidInfo{12, 20, 30}
};

class Engine {
public:
    Engine()
        : scene(&solidManager),
          camera(glm::vec3(0,0,3),
                 glm::vec3(0,0,0),
                 glm::vec3(0,1,0))
    {
        initScene();
        initUI();

        int w, h;
        glfwGetFramebufferSize(graphics.getWindow(), &w, &h);

        if (w != lastWidth || h != lastHeight) {
            initUI();
            lastWidth = w;
            lastHeight = h;
        }

        camera.addObserver(&consoleObserver);

        setupCallbacks();
        graphics.setupShadowMap(1024, 1024);
    }

    void run() {
        while (!graphics.shouldClose()) {
            float dt = graphics.beginFrame();

            update(dt);

            graphics.prepareFrame();

            RenderContext ctx;
            ctx.showInfoWindow = showInfoWindow;
            ctx.fps = currentFPS;
            ctx.showFPS = showFPS;
            ctx.activeIndex = solidManager.getActiveIndex();
            ctx.view = camera.getViewMatrix();
            ctx.projection = camera.getProjectionMatrix(graphics.getAspectRatio());
            ctx.cameraPos = camera.getPosition();
            ctx.lightPos = glm::vec3(2.0f, 2.0f, 2.0f);
            ctx.hoveredIndex = hoveredIndex;

            auto obj = solidManager.getActiveObject();

            if (obj) {
                std::shared_ptr<PlatonicSolid> ps;

                auto wire = std::dynamic_pointer_cast<WireframeDecorator>(obj);
                if (wire) {
                    ps = std::dynamic_pointer_cast<PlatonicSolid>(wire->getComponent());
                } else {
                    ps = std::dynamic_pointer_cast<PlatonicSolid>(obj);
                }

                if (ps) {
                    ctx.verticesCount = ps->getVertexCount();
                    ctx.facesCount = ps->getFaceCount();
                    ctx.edgesCount = ps->getEdgeCount();
                }
            }

            graphics.render(scene, ctx);

            graphics.finalizeFrame();
        }
    }
    

private:
    std::shared_ptr<IShaderProvider> shaderProvider =
    std::make_shared<DefaultShaderProvider>();

    GraphicsSystem graphics{shaderProvider};
    InputSystem inputSystem;

    SolidManager solidManager;
    Scene scene;
    Camera camera;
    ConsoleObserver consoleObserver;

    bool showFPS = false;
    bool viewModeAutoRotate = false;
    bool showInfoWindow = false;
    double fpsTimer = 0.0;
    int fpsFrames = 0;
    int currentFPS = 0;

    float idleTime = 0.0f;
    const float idleThreshold = 90.0f;
    const float idleRotateSpeed = 20.0f;

    void initScene() {
        solidManager.addObserver(&consoleObserver);

        solidManager.addObject(std::make_shared<Cube>());

        solidManager.addObject(std::make_shared<Tetrahedron>());

        solidManager.addObject(std::make_shared<Octahedron>());

        solidManager.addObject(std::make_shared<Icosahedron>());

        solidManager.addObject(std::make_shared<Dodecahedron>());
    }

    void initUI() {
        buttons.clear();

        int w, h;
        glfwGetFramebufferSize(graphics.getWindow(), &w, &h);

        float panelWidth = w * 0.12f;
        float buttonSize = panelWidth * 0.5f;
        float spacing = panelWidth * 0.15f;

        float startY = h - buttonSize - spacing;

        for (int i = 0; i < 5; i++) {
            float y = startY - i * (buttonSize + spacing);
            float x = w - panelWidth + (panelWidth - buttonSize) / 2.0f;

            float textHeight = buttonSize * 0.3f;
            float paddingTop = buttonSize * 0.5f;

            float clickY = y + paddingTop;

            buttons.push_back({
                x,
                clickY,
                buttonSize,
                textHeight
            });
        }

        float lastButtonY = startY - 4 * (buttonSize + spacing);

        {
            float textHeight = buttonSize * 0.3f;
            float paddingTop = 0.0f;

            float infoY = lastButtonY - spacing * 2.0f;

            buttons.push_back({
                w - panelWidth,
                infoY,
                panelWidth,
                textHeight
            });
        }

        {
            float textHeight = buttonSize * 0.3f;

            float viewY = lastButtonY - spacing * 2.0f - spacing * 4.0f;

            buttons.push_back({
                w - panelWidth,
                viewY,
                panelWidth,
                textHeight
            });
        }
    }

    void setupCallbacks() {
        GLFWwindow* window = graphics.getWindow();

        glfwSetWindowUserPointer(window, &inputSystem);

        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetKeyCallback(window, keyCallback);
    }

    void update(float dt) {
        int w, h;
        glfwGetFramebufferSize(graphics.getWindow(), &w, &h);

        if (w != lastWidth || h != lastHeight) {
            initUI();
            lastWidth = w;
            lastHeight = h;
        }

        InputState& input = inputSystem.getState();

        updateCamera(input, dt);
        updateScene(input);
        updateFPS(dt);
    }

    void updateCamera(InputState& input, float dt) {
            bool isUserActive =
        input.mouseDeltaX != 0.0f ||
        input.mouseDeltaY != 0.0f ||
        input.scrollDelta != 0.0f ||
        input.leftKeyPressed ||
        input.rightKeyPressed ||
        input.upKeyPressed ||
        input.downKeyPressed;

        const float mouseRotateSpeed = 0.25f;
        const float scrollZoomStep = 0.3f;
        const float keyRotateSpeed = 120.0f;
        const float keyZoomSpeed = 4.0f;

        if (input.mouseDeltaX != 0.0f || input.mouseDeltaY != 0.0f) {
            camera.rotate(-input.mouseDeltaX * mouseRotateSpeed,
                          input.mouseDeltaY * mouseRotateSpeed);
        }

        if (input.scrollDelta != 0.0f) {
            camera.zoom(-input.scrollDelta * scrollZoomStep);
        }

        if (input.leftKeyPressed) {
            camera.rotate(-keyRotateSpeed * dt, 0.0f);
        }
        if (input.rightKeyPressed) {
            camera.rotate(keyRotateSpeed * dt, 0.0f);
        }
        if (input.upKeyPressed) {
            camera.zoom(-keyZoomSpeed * dt);
        }
        if (input.downKeyPressed) {
            camera.zoom(keyZoomSpeed * dt);
        }

        if (isUserActive) {
            idleTime = 0.0f;
            viewModeAutoRotate = false;
        } else {
            idleTime += dt;
        }

        if (viewModeAutoRotate || idleTime >= idleThreshold) {
            camera.rotate(idleRotateSpeed * dt, 0.0f);
        }

        input.mouseDeltaX = 0.0f;
        input.mouseDeltaY = 0.0f;
        input.scrollDelta = 0.0f;
    }

    void updateScene(InputState& input) {
        for (int i = 0; i < 5; ++i) {
            if (!input.numberKeyPressed[i]) continue;

            solidManager.setActiveIndex(i);
            input.numberKeyPressed[i] = false;
        }

        if (input.toggleWireframe) {
            int index = solidManager.getActiveIndex();
            auto& objects = solidManager.getObjectsMutable();

            if (index >= 0 && index < (int)objects.size()) {
                auto current = objects[index];

                auto wire = std::dynamic_pointer_cast<WireframeDecorator>(current);

                if (wire) {
                    objects[index] = wire->getComponent();
                } else {
                    objects[index] = std::make_shared<WireframeDecorator>(current);
                }
            }

            input.toggleWireframe = false;
        }

        if (input.toggleFPS) {
            showFPS = !showFPS;

            if (!showFPS) {
                glfwSetWindowTitle(graphics.getWindow(), "Platon Engine");
            }

            input.toggleFPS = false;
        }
        if (input.mouseClicked) {
            double mx = input.mouseX;
            double my = input.mouseY;

            int w, h;
            glfwGetFramebufferSize(graphics.getWindow(), &w, &h);

            my = h - my;

            for (int i = 0; i < static_cast<int>(buttons.size()); i++) {
                const auto& b = buttons[i];

                if (mx >= b.x && mx <= b.x + b.width &&
                    my >= b.y && my <= b.y + b.height)
                {
                    if (i < 5) {
                        solidManager.setActiveIndex(i);
                    }
                    else if (i == 5) {
                        showInfoWindow = !showInfoWindow;
                    }
                    else if (i == 6) {
                        viewModeAutoRotate = !viewModeAutoRotate;

                        std::cout << "VIEW MODE toggled: "
                                << (viewModeAutoRotate ? "ON" : "OFF")
                                << std::endl;
                    }

                    break;
                }
            }

            input.mouseClicked = false;
        }

        hoveredIndex = -1;

        double mx = input.mouseX;
        double my = input.mouseY;

        int w, h;
        glfwGetFramebufferSize(graphics.getWindow(), &w, &h);

        my = h - my;

        for (int i = 0; i < static_cast<int>(buttons.size()); i++) {
            const auto& b = buttons[i];

            if (mx >= b.x && mx <= b.x + b.width &&
                my >= b.y && my <= b.y + b.height)
            {
                hoveredIndex = i;
                break;
            }
        }
    }

    void updateFPS(float dt) {
        if (!showFPS) {
            return;
        }

        fpsTimer += dt;
        fpsFrames++;

        if (fpsTimer >= 0.5) {
            currentFPS = static_cast<int>(fpsFrames / fpsTimer);

            fpsTimer = 0.0;
            fpsFrames = 0;
        }
    }

    std::vector<UIButton> buttons;
    int lastWidth = 0;
    int lastHeight = 0;

    int hoveredIndex = -1;
};

int main() {
    try {
        Engine engine;
        engine.run();

        ShaderManager::getInstance().releaseAll();
        return 0;
    } catch (const std::exception& e) {
        std::cout << e.what() << '\n';
        return -1;
    }
}