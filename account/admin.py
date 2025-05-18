from django.contrib import admin
from django.contrib.auth.models import Group
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser
from .forms import RegistrationForm

# CustomUser Admin Registration
class CustomUserAdmin(UserAdmin):
  
    model = CustomUser
    list_display = ['username', 'email', 'role', 'is_staff', 'is_active']
    list_filter = ['role', 'is_staff', 'is_active']
    search_fields = ['username', 'email']
    ordering = ['username']
    
    fieldsets = (
        (None, {'fields': ('username', 'email', 'password')}),  # Basic user fields
        ('Personal info', {'fields': ('first_name', 'last_name', 'role')}),  # Include role in personal info
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),  # Permissions
        ('Important dates', {'fields': ('last_login', 'date_joined')}),  # Important dates
    )

    add_fieldsets = (
        (None, {'fields': ('username', 'email', 'password1', 'password2', 'role')}),  # Include role during user creation
    )

    filter_horizontal = ('groups', 'user_permissions')

    def save_model(self, request, obj, form, change):
        obj.save()
        
        # Automatically assign the user to the corresponding group based on role
        if obj.role:
            group, created = Group.objects.get_or_create(name=obj.role.capitalize())  # Ensure the group name is properly capitalized
            obj.groups.add(group)
        
        obj.save()

admin.site.register(CustomUser, CustomUserAdmin)