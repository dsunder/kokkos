INCLUDE(CMakeParseArguments)
INCLUDE(CTest)

cmake_policy(SET CMP0054 NEW)

MESSAGE(STATUS "The project name is: ${PROJECT_NAME}")

IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_OpenMP)
  SET(${PROJECT_NAME}_ENABLE_OpenMP OFF)
ENDIF()

IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_DEBUG)
  SET(${PROJECT_NAME}_ENABLE_DEBUG OFF)
ENDIF()

IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_CXX11)
  SET(${PROJECT_NAME}_ENABLE_CXX11 ON)
ENDIF()

IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_TESTS)
  SET(${PROJECT_NAME}_ENABLE_TESTS OFF)
ENDIF()

IF(NOT DEFINED TPL_ENABLE_Pthread)
  SET(TPL_ENABLE_Pthread OFF)
ENDIF()

FUNCTION(ASSERT_DEFINED VARS)
  FOREACH(VAR ${VARS})
    IF(NOT DEFINED ${VAR})
      MESSAGE(SEND_ERROR "Error, the variable ${VAR} is not defined!")
    ENDIF()
  ENDFOREACH()
ENDFUNCTION()

MACRO(GLOBAL_SET VARNAME)
  SET(${VARNAME} ${ARGN} CACHE INTERNAL "")
ENDMACRO()

MACRO(PREPEND_GLOBAL_SET VARNAME)
  ASSERT_DEFINED(${VARNAME})
  GLOBAL_SET(${VARNAME} ${ARGN} ${${VARNAME}})
ENDMACRO()

#FUNCTION(REMOVE_GLOBAL_DUPLICATES VARNAME)
#  ASSERT_DEFINED(${VARNAME})
#  IF (${VARNAME})
#    SET(TMP ${${VARNAME}})
#    LIST(REMOVE_DUPLICATES TMP)
#    GLOBAL_SET(${VARNAME} ${TMP})
#  ENDIF()
#ENDFUNCTION()

#MACRO(TRIBITS_ADD_OPTION_AND_DEFINE  USER_OPTION_NAME  MACRO_DEFINE_NAME DOCSTRING  DEFAULT_VALUE)
#  MESSAGE(STATUS "TRIBITS_ADD_OPTION_AND_DEFINE: '${USER_OPTION_NAME}' '${MACRO_DEFINE_NAME}' '${DEFAULT_VALUE}'")
#  SET( ${USER_OPTION_NAME} "${DEFAULT_VALUE}" CACHE BOOL "${DOCSTRING}" )
#  IF(NOT ${MACRO_DEFINE_NAME} STREQUAL "")
#    IF(${USER_OPTION_NAME})
#      GLOBAL_SET(${MACRO_DEFINE_NAME} ON)
#    ELSE()
#      GLOBAL_SET(${MACRO_DEFINE_NAME} OFF)
#    ENDIF()
#  ENDIF()
#ENDMACRO()

FUNCTION(TRIBITS_CONFIGURE_FILE  PACKAGE_NAME_CONFIG_FILE)

  # Configure the file
  CONFIGURE_FILE(
    ${PACKAGE_SOURCE_DIR}/cmake/${PACKAGE_NAME_CONFIG_FILE}.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME_CONFIG_FILE}
    )

ENDFUNCTION()

#MACRO(TRIBITS_ADD_DEBUG_OPTION)
#  TRIBITS_ADD_OPTION_AND_DEFINE(
#    ${PROJECT_NAME}_ENABLE_DEBUG
#    HAVE_${PROJECT_NAME_UC}_DEBUG
#    "Enable a host of runtime debug checking."
#    OFF
#    )
#ENDMACRO()


MACRO(TRIBITS_ADD_TEST_DIRECTORIES)
  message(STATUS "ProjectName: " ${PROJECT_NAME})
  message(STATUS "Tests: " ${${PROJECT_NAME}_ENABLE_TESTS})
  
  IF(${${PROJECT_NAME}_ENABLE_TESTS})
    FOREACH(TEST_DIR ${ARGN})
      ADD_SUBDIRECTORY(${TEST_DIR})
    ENDFOREACH()
  ENDIF()
ENDMACRO()

MACRO(TRIBITS_ADD_EXAMPLE_DIRECTORIES)

  IF(${PACKAGE_NAME}_ENABLE_EXAMPLES OR ${PARENT_PACKAGE_NAME}_ENABLE_EXAMPLES)
    FOREACH(EXAMPLE_DIR ${ARGN})
      ADD_SUBDIRECTORY(${EXAMPLE_DIR})
    ENDFOREACH()
  ENDIF()

ENDMACRO()


function(INCLUDE_DIRECTORIES)
  cmake_parse_arguments(INCLUDE_DIRECTORIES "REQUIRED_DURING_INSTALLATION_TESTING" "" "" ${ARGN})
  _INCLUDE_DIRECTORIES(${INCLUDE_DIRECTORIES_UNPARSED_ARGUMENTS})
endfunction()


MACRO(TARGET_TRANSFER_PROPERTY TARGET_NAME PROP_IN PROP_OUT)
  SET(PROP_VALUES)
  FOREACH(TARGET_X ${ARGN})
    LIST(APPEND PROP_VALUES "$<TARGET_PROPERTY:${TARGET_X},${PROP_IN}>")
  ENDFOREACH()
  SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES ${PROP_OUT} "${PROP_VALUES}")
ENDMACRO()

MACRO(ADD_INTERFACE_LIBRARY LIB_NAME)
  FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "")
  ADD_LIBRARY(${LIB_NAME} STATIC ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
  SET_TARGET_PROPERTIES(${LIB_NAME} PROPERTIES INTERFACE TRUE)
ENDMACRO()

# Older versions of cmake does not make include directories transitive
MACRO(TARGET_LINK_AND_INCLUDE_LIBRARIES TARGET_NAME)
  TARGET_LINK_LIBRARIES(${TARGET_NAME} LINK_PUBLIC ${ARGN})
  FOREACH(DEP_LIB ${ARGN})
    TARGET_INCLUDE_DIRECTORIES(${TARGET_NAME} PUBLIC $<TARGET_PROPERTY:${DEP_LIB},INTERFACE_INCLUDE_DIRECTORIES>)
    TARGET_INCLUDE_DIRECTORIES(${TARGET_NAME} PUBLIC $<TARGET_PROPERTY:${DEP_LIB},INCLUDE_DIRECTORIES>)
  ENDFOREACH()
ENDMACRO()

FUNCTION(TRIBITS_ADD_LIBRARY LIBRARY_NAME)

  SET(options STATIC SHARED TESTONLY NO_INSTALL_LIB_OR_HEADERS CUDALIBRARY)
  SET(oneValueArgs)
  SET(multiValueArgs HEADERS HEADERS_INSTALL_SUBDIR NOINSTALLHEADERS SOURCES DEPLIBS IMPORTEDLIBS DEFINES ADDED_LIB_TARGET_NAME_OUT)

  CMAKE_PARSE_ARGUMENTS(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  IF(PARSE_HEADERS)
    LIST(REMOVE_DUPLICATES PARSE_HEADERS)
  ENDIF()
  IF(PARSE_SOURCES)
    LIST(REMOVE_DUPLICATES PARSE_SOURCES)
  ENDIF()

  # Local variable to hold all of the libraries that will be directly linked
  # to this library.
  SET(LINK_LIBS ${${PACKAGE_NAME}_DEPS})

  # Add dependent libraries passed directly in

  IF (PARSE_IMPORTEDLIBS)
    LIST(APPEND LINK_LIBS ${PARSE_IMPORTEDLIBS})
  ENDIF()

  IF (PARSE_DEPLIBS)
    LIST(APPEND LINK_LIBS ${PARSE_DEPLIBS})
  ENDIF()

  # Add the library and all the dependencies

  IF (PARSE_DEFINES)
    ADD_DEFINITIONS(${PARSE_DEFINES})
  ENDIF()

  IF (PARSE_STATIC)
    SET(STATIC_KEYWORD "STATIC")
  ELSE()
    SET(STATIC_KEYWORD)
  ENDIF()

  IF (PARSE_SHARED)
    SET(SHARED_KEYWORD "SHARED")
  ELSE()
    SET(SHARED_KEYWORD)
  ENDIF()

  IF (PARSE_TESTONLY)
    SET(EXCLUDE_FROM_ALL_KEYWORD "EXCLUDE_FROM_ALL")
  ELSE()
    SET(EXCLUDE_FROM_ALL_KEYWORD)
  ENDIF()
  IF (NOT PARSE_CUDALIBRARY)
    ADD_LIBRARY(
      ${LIBRARY_NAME}
      ${STATIC_KEYWORD}
      ${SHARED_KEYWORD}
      ${EXCLUDE_FROM_ALL_KEYWORD}
      ${PARSE_HEADERS}
      ${PARSE_NOINSTALLHEADERS}
      ${PARSE_SOURCES}
      )
  ELSE()
    CUDA_ADD_LIBRARY(
      ${LIBRARY_NAME}
      ${PARSE_HEADERS}
      ${PARSE_NOINSTALLHEADERS}
      ${PARSE_SOURCES}
      )
  ENDIF()

  TARGET_LINK_AND_INCLUDE_LIBRARIES(${LIBRARY_NAME} ${LINK_LIBS})

  IF (NOT PARSE_TESTONLY OR PARSE_NO_INSTALL_LIB_OR_HEADERS)

    INSTALL(
      TARGETS ${LIBRARY_NAME}
      EXPORT ${PROJECT_NAME}
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      COMPONENT ${PACKAGE_NAME}
      )

    INSTALL(
      FILES  ${PARSE_HEADERS}
      EXPORT ${PROJECT_NAME}
      DESTINATION include
      COMPONENT ${PACKAGE_NAME}
      )

      INSTALL(
      DIRECTORY  ${PARSE_HEADERS_INSTALL_SUBDIR}
      EXPORT ${PROJECT_NAME}
      DESTINATION include
      COMPONENT ${PACKAGE_NAME}
      )

  ENDIF()

  IF (NOT PARSE_TESTONLY)
    PREPEND_GLOBAL_SET(${PACKAGE_NAME}_LIBS ${LIBRARY_NAME})
    REMOVE_GLOBAL_DUPLICATES(${PACKAGE_NAME}_LIBS)
  ENDIF()

ENDFUNCTION()

FUNCTION(TRIBITS_ADD_EXECUTABLE EXE_NAME)

  SET(options NOEXEPREFIX NOEXESUFFIX ADD_DIR_TO_NAME INSTALLABLE TESTONLY)
  SET(oneValueArgs ADDED_EXE_TARGET_NAME_OUT)
  SET(multiValueArgs SOURCES CATEGORIES HOST XHOST HOSTTYPE XHOSTTYPE DIRECTORY TESTONLYLIBS IMPORTEDLIBS DEPLIBS COMM LINKER_LANGUAGE TARGET_DEFINES DEFINES)

  CMAKE_PARSE_ARGUMENTS(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  IF (PARSE_TARGET_DEFINES)
    TARGET_COMPILE_DEFINITIONS(${EXE_NAME} PUBLIC ${PARSE_TARGET_DEFINES})
  ENDIF()

  SET(LINK_LIBS PACKAGE_${PACKAGE_NAME})

  IF (PARSE_TESTONLYLIBS)
    LIST(APPEND LINK_LIBS ${PARSE_TESTONLYLIBS})
  ENDIF()

  IF (PARSE_IMPORTEDLIBS)
    LIST(APPEND LINK_LIBS ${PARSE_IMPORTEDLIBS})
  ENDIF()

  SET (EXE_SOURCES)
  IF(PARSE_DIRECTORY)
    FOREACH( SOURCE_FILE ${PARSE_SOURCES} )
      IF(IS_ABSOLUTE ${SOURCE_FILE})
        SET (EXE_SOURCES ${EXE_SOURCES} ${SOURCE_FILE})
      ELSE()
        SET (EXE_SOURCES ${EXE_SOURCES} ${PARSE_DIRECTORY}/${SOURCE_FILE})
      ENDIF()
    ENDFOREACH( )
  ELSE()
    FOREACH( SOURCE_FILE ${PARSE_SOURCES} )
      SET (EXE_SOURCES ${EXE_SOURCES} ${SOURCE_FILE})
    ENDFOREACH( )
  ENDIF()

  SET(EXE_BINARY_NAME ${EXE_NAME})
  IF(DEFINED PACKAGE_NAME AND NOT PARSE_NOEXEPREFIX)
    SET(EXE_BINARY_NAME ${PACKAGE_NAME}_${EXE_BINARY_NAME})
  ENDIF()

  # IF (PARSE_TESTONLY)
  #   SET(EXCLUDE_FROM_ALL_KEYWORD "EXCLUDE_FROM_ALL")
  # ELSE()
  #   SET(EXCLUDE_FROM_ALL_KEYWORD)
  # ENDIF()
  ADD_EXECUTABLE(${EXE_BINARY_NAME} ${EXCLUDE_FROM_ALL_KEYWORD} ${EXE_SOURCES})

  TARGET_LINK_AND_INCLUDE_LIBRARIES(${EXE_BINARY_NAME} ${LINK_LIBS})

  IF(PARSE_ADDED_EXE_TARGET_NAME_OUT)
    SET(${PARSE_ADDED_EXE_TARGET_NAME_OUT} ${EXE_BINARY_NAME} PARENT_SCOPE)
  ENDIF()

  IF(PARSE_INSTALLABLE)
    INSTALL(
      TARGETS ${EXE_BINARY_NAME}
      EXPORT ${PROJECT_NAME}
        DESTINATION bin
    )
  ENDIF()
ENDFUNCTION()

ADD_CUSTOM_TARGET(check COMMAND ${CMAKE_CTEST_COMMAND} -VV -C ${CMAKE_CFG_INTDIR})

FUNCTION(TRIBITS_ADD_TEST)
ENDFUNCTION()
FUNCTION(TRIBITS_TPL_TENTATIVELY_ENABLE)
ENDFUNCTION()

FUNCTION(TRIBITS_ADD_EXECUTABLE_AND_TEST EXE_NAME)

  SET(options STANDARD_PASS_OUTPUT WILL_FAIL)
  SET(oneValueArgs PASS_REGULAR_EXPRESSION FAIL_REGULAR_EXPRESSION ENVIRONMENT TIMEOUT CATEGORIES ADDED_TESTS_NAMES_OUT ADDED_EXE_TARGET_NAME_OUT)
  SET(multiValueArgs)

  CMAKE_PARSE_ARGUMENTS(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  TRIBITS_ADD_EXECUTABLE(${EXE_NAME} TESTONLY ADDED_EXE_TARGET_NAME_OUT TEST_NAME ${PARSE_UNPARSED_ARGUMENTS})

  IF(WIN32)
    ADD_TEST(NAME ${TEST_NAME} WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH} COMMAND ${TEST_NAME}${CMAKE_EXECUTABLE_SUFFIX})
  ELSE()
    ADD_TEST(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  ENDIF()
  ADD_DEPENDENCIES(check ${TEST_NAME})

  IF(PARSE_FAIL_REGULAR_EXPRESSION)
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION ${PARSE_FAIL_REGULAR_EXPRESSION})
  ENDIF()

  IF(PARSE_PASS_REGULAR_EXPRESSION)
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES PASS_REGULAR_EXPRESSION ${PARSE_PASS_REGULAR_EXPRESSION})
  ENDIF()

  IF(PARSE_WILL_FAIL)
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES WILL_FAIL ${PARSE_WILL_FAIL})
  ENDIF()

  IF(PARSE_ADDED_TESTS_NAMES_OUT)
    SET(${PARSE_ADDED_TESTS_NAMES_OUT} ${TEST_NAME} PARENT_SCOPE)
  ENDIF()

  IF(PARSE_ADDED_EXE_TARGET_NAME_OUT)
    SET(${PARSE_ADDED_EXE_TARGET_NAME_OUT} ${TEST_NAME} PARENT_SCOPE)
  ENDIF()

ENDFUNCTION()

MACRO(TIBITS_CREATE_IMPORTED_TPL_LIBRARY TPL_NAME)
  ADD_INTERFACE_LIBRARY(TPL_LIB_${TPL_NAME})
  TARGET_LINK_LIBRARIES(TPL_LIB_${TPL_NAME} LINK_PUBLIC ${TPL_${TPL_NAME}_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(TPL_LIB_${TPL_NAME} INTERFACE ${TPL_${TPL_NAME}_INCLUDE_DIRS})
ENDMACRO()

FUNCTION(TRIBITS_TPL_FIND_INCLUDE_DIRS_AND_LIBRARIES TPL_NAME)

  SET(options MUST_FIND_ALL_LIBS MUST_FIND_ALL_HEADERS NO_PRINT_ENABLE_SUCCESS_FAIL)
  SET(oneValueArgs)
  SET(multiValueArgs REQUIRED_HEADERS REQUIRED_LIBS_NAMES)

  CMAKE_PARSE_ARGUMENTS(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  SET(_${TPL_NAME}_ENABLE_SUCCESS TRUE)
  IF (PARSE_REQUIRED_LIBS_NAMES)
    FIND_LIBRARY(TPL_${TPL_NAME}_LIBRARIES NAMES ${PARSE_REQUIRED_LIBS_NAMES})
    IF(NOT TPL_${TPL_NAME}_LIBRARIES)
      SET(_${TPL_NAME}_ENABLE_SUCCESS FALSE)
    ENDIF()
  ENDIF()
  IF (PARSE_REQUIRED_HEADERS)
    FIND_PATH(TPL_${TPL_NAME}_INCLUDE_DIRS NAMES ${PARSE_REQUIRED_HEADERS})
    IF(NOT TPL_${TPL_NAME}_INCLUDE_DIRS)
      SET(_${TPL_NAME}_ENABLE_SUCCESS FALSE)
    ENDIF()
  ENDIF()


  IF (_${TPL_NAME}_ENABLE_SUCCESS)
    TIBITS_CREATE_IMPORTED_TPL_LIBRARY(${TPL_NAME})
  ENDIF()

ENDFUNCTION()

#MACRO(TRIBITS_PROCESS_TPL_DEP_FILE TPL_FILE)
#  GET_FILENAME_COMPONENT(TPL_NAME ${TPL_FILE} NAME_WE)
#  INCLUDE("${TPL_FILE}")
#  IF(TARGET TPL_LIB_${TPL_NAME})
#    MESSAGE(STATUS "Found tpl library: ${TPL_NAME}")
#    SET(TPL_ENABLE_${TPL_NAME} TRUE)
#  ELSE()
#    MESSAGE(STATUS "Tpl library not found: ${TPL_NAME}")
#    SET(TPL_ENABLE_${TPL_NAME} FALSE)
#  ENDIF()
#ENDMACRO()

MACRO(PREPEND_TARGET_SET VARNAME TARGET_NAME TYPE)
  IF(TYPE STREQUAL "REQUIRED")
    SET(REQUIRED TRUE)
  ELSE()
    SET(REQUIRED FALSE)
  ENDIF()
  IF(TARGET ${TARGET_NAME})
    PREPEND_GLOBAL_SET(${VARNAME} ${TARGET_NAME})
  ELSE()
    IF(REQUIRED)
      MESSAGE(FATAL_ERROR "Missing dependency ${TARGET_NAME}")
    ENDIF()
  ENDIF()
ENDMACRO()

MACRO(TRIBITS_APPEND_PACKAGE_DEPS DEP_LIST TYPE)
  FOREACH(DEP ${ARGN})
    PREPEND_GLOBAL_SET(${DEP_LIST} PACKAGE_${DEP})
  ENDFOREACH()
ENDMACRO()

MACRO(TRIBITS_APPEND_TPLS_DEPS DEP_LIST TYPE)
  FOREACH(DEP ${ARGN})
    PREPEND_TARGET_SET(${DEP_LIST} TPL_LIB_${DEP} ${TYPE})
  ENDFOREACH()
ENDMACRO()

MACRO(TRIBITS_ENABLE_TPLS)
  FOREACH(TPL ${ARGN})
    IF(TARGET ${TPL})
      GLOBAL_SET(${PACKAGE_NAME}_ENABLE_${TPL} TRUE)
    ELSE()
      GLOBAL_SET(${PACKAGE_NAME}_ENABLE_${TPL} FALSE)
    ENDIF()
  ENDFOREACH()
ENDMACRO()

MACRO(TRIBITS_PACKAGE_DEFINE_DEPENDENCIES)

  SET(options)
  SET(oneValueArgs)
  SET(multiValueArgs 
    LIB_REQUIRED_PACKAGES
    LIB_OPTIONAL_PACKAGES
    TEST_REQUIRED_PACKAGES
    TEST_OPTIONAL_PACKAGES
    LIB_REQUIRED_TPLS
    LIB_OPTIONAL_TPLS
    TEST_REQUIRED_TPLS
    TEST_OPTIONAL_TPLS
    REGRESSION_EMAIL_LIST
    SUBPACKAGES_DIRS_CLASSIFICATIONS_OPTREQS
  )
  CMAKE_PARSE_ARGUMENTS(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  GLOBAL_SET(${PACKAGE_NAME}_DEPS "")
  TRIBITS_APPEND_PACKAGE_DEPS(${PACKAGE_NAME}_DEPS REQUIRED ${PARSE_LIB_REQUIRED_PACKAGES})
  TRIBITS_APPEND_PACKAGE_DEPS(${PACKAGE_NAME}_DEPS OPTIONAL ${PARSE_LIB_OPTIONAL_PACKAGES})
  TRIBITS_APPEND_TPLS_DEPS(${PACKAGE_NAME}_DEPS REQUIRED ${PARSE_LIB_REQUIRED_TPLS})
  TRIBITS_APPEND_TPLS_DEPS(${PACKAGE_NAME}_DEPS OPTIONAL ${PARSE_LIB_OPTIONAL_TPLS})

  GLOBAL_SET(${PACKAGE_NAME}_TEST_DEPS "")
  TRIBITS_APPEND_PACKAGE_DEPS(${PACKAGE_NAME}_TEST_DEPS REQUIRED ${PARSE_TEST_REQUIRED_PACKAGES})
  TRIBITS_APPEND_PACKAGE_DEPS(${PACKAGE_NAME}_TEST_DEPS OPTIONAL ${PARSE_TEST_OPTIONAL_PACKAGES})
  TRIBITS_APPEND_TPLS_DEPS(${PACKAGE_NAME}_TEST_DEPS REQUIRED ${PARSE_TEST_REQUIRED_TPLS})
  TRIBITS_APPEND_TPLS_DEPS(${PACKAGE_NAME}_TEST_DEPS OPTIONAL ${PARSE_TEST_OPTIONAL_TPLS})

  TRIBITS_ENABLE_TPLS(${PARSE_LIB_REQUIRED_TPLS} ${PARSE_LIB_OPTIONAL_TPLS} ${PARSE_TEST_REQUIRED_TPLS} ${PARSE_TEST_OPTIONAL_TPLS})

ENDMACRO()

MACRO(TRIBITS_SUBPACKAGE NAME)
  SET(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  SET(PARENT_PACKAGE_NAME ${PACKAGE_NAME})
  SET(PACKAGE_NAME ${PACKAGE_NAME}${NAME})
  STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  ADD_INTERFACE_LIBRARY(PACKAGE_${PACKAGE_NAME})

  GLOBAL_SET(${PACKAGE_NAME}_LIBS "")

  INCLUDE(${PACKAGE_SOURCE_DIR}/cmake/Dependencies.cmake)

ENDMACRO(TRIBITS_SUBPACKAGE)

MACRO(TRIBITS_SUBPACKAGE_POSTPROCESS)
  TARGET_LINK_AND_INCLUDE_LIBRARIES(PACKAGE_${PACKAGE_NAME} ${${PACKAGE_NAME}_LIBS})
ENDMACRO(TRIBITS_SUBPACKAGE_POSTPROCESS)

MACRO(TRIBITS_PACKAGE_DECL NAME)

  SET(PACKAGE_NAME ${NAME})
  SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)

  #SET(TRIBITS_DEPS_DIR "${CMAKE_SOURCE_DIR}/cmake/deps")
  #FILE(GLOB TPLS_FILES "${TRIBITS_DEPS_DIR}/*.cmake")
  #FOREACH(TPL_FILE ${TPLS_FILES})
  #  TRIBITS_PROCESS_TPL_DEP_FILE(${TPL_FILE})
  #ENDFOREACH()

ENDMACRO()


MACRO(TRIBITS_PROCESS_SUBPACKAGES)
  FILE(GLOB SUBPACKAGES RELATIVE ${CMAKE_SOURCE_DIR} */cmake/Dependencies.cmake)
  FOREACH(SUBPACKAGE ${SUBPACKAGES})
    GET_FILENAME_COMPONENT(SUBPACKAGE_CMAKE ${SUBPACKAGE} DIRECTORY)
    GET_FILENAME_COMPONENT(SUBPACKAGE_DIR ${SUBPACKAGE_CMAKE} DIRECTORY)
    ADD_SUBDIRECTORY(${CMAKE_BINARY_DIR}/../${SUBPACKAGE_DIR})
  ENDFOREACH()
ENDMACRO(TRIBITS_PROCESS_SUBPACKAGES)

MACRO(TRIBITS_PACKAGE_DEF)
ENDMACRO(TRIBITS_PACKAGE_DEF)

MACRO(TRIBITS_EXCLUDE_AUTOTOOLS_FILES)
ENDMACRO(TRIBITS_EXCLUDE_AUTOTOOLS_FILES)

MACRO(TRIBITS_EXCLUDE_FILES)
ENDMACRO(TRIBITS_EXCLUDE_FILES)

MACRO(TRIBITS_PACKAGE_POSTPROCESS)
ENDMACRO(TRIBITS_PACKAGE_POSTPROCESS)

