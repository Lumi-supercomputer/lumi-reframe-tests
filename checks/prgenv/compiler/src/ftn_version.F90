program check_fortran_version
#ifdef _CRAYFTN
  write (*,*) _RELEASE_STRING
#elif __FLANG
! Find better flang version string macro 
  write (*,*) __FLANG_MAJOR__ 
#else
  write (*,*) __VERSION__
#endif
end program check_fortran_version
