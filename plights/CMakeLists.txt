
file(GLOB subdirectories LIST_DIRECTORIES true "*")
foreach(directory ${subdirectories})
    if(IS_DIRECTORY ${directory})
        add_subdirectory(${directory})
    endif()
endforeach()
