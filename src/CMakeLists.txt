set(SOURCES
        gcmma.cpp)

set(HEADERS
        gcmma.h)

add_library(${CMAKE_PROJECT_NAME}_lib STATIC ${SOURCES} ${HEADERS})